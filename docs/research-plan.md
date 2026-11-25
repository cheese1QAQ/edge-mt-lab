# Edge-MT-Lab 研究总方案（RTX 3060 8GB / 2~4 GB LLM 预算）

## 1. 硬件物理边界与预算拆解

本项目以消费级中低端显卡 **NVIDIA RTX 3060 8GB（Ampere SM86）** 为核心约束基准：

| 维度 | RTX 3060 8GB 现实特性 | 对研究技术选型的直接约束 |
| :--- | :--- | :--- |
| **显存容量** | 8 GB 总显存（需与 OS、CUDA Context、OCR 共存） | LLM **总显存预算严控在 2.0~4.0 GB**，留给权重的安全空间约 **1.5~3.2 GB** |
| **显存位宽与带宽** | 128-bit，约 240 GB/s（低于 12GB 版的 360 GB/s） | Decode 阶段严格 **Memory-Bandwidth Bound**：权重/KV 每压缩 50%，生成速度近乎翻倍 |
| **Tensor Core 特性** | 原生支持 FP16 / BF16 / INT8 / 2:4 稀疏 / `cp.async` | **不支持 Hopper/Blackwell 原生 FP8/NVFP4**；优先选 **Marlin (W4A16)**、**BitBLAS / LUT-GEMM** 与 `llama.cpp` |
| **多框架共存开销** | 每个独立 PyTorch/CUDA 进程约占 0.5~1.0 GB 固定上下文 | 尽量让轻量化 OCR / Embedding 驻留 **CPU ONNX Runtime**，或将视觉与翻译统一到单一推理进程内 |

### 典型 8 GB 整卡显存分档目标

* **2.0 GB 极限档**：`1.5B–3B @ 4-bit` 或 `4B @ 2~3-bit` 或 `BitNet b1.58 2B4T (~0.5 GB)` + 强外部 RAG/术语补偿。
* **3.0 GB 均衡档（主力甜点）**：`Qwen3-4B @ 4-bit (词表裁剪)` vs `Qwen3-8B @ 2.5-bit (QTIP / Dynamic 2.0)` + `KV Q4/MLA`。
* **4.0 GB 高质档**：`Qwen3-4B @ Q5/Q6 + 紧凑 VLM` 或 `Qwen3-8B / Seed-X-7B @ 3~3.5-bit` + 独立 0.9B 级轻量 OCR VLM（如 PaddleOCR-VL）。

---

## 2. 四大递进式研究主线

### Track 1：比特–规模缩放律与前沿权重量化（`experiments/tracks/01-quantization/`）
* **研究问题**：在严格 2.0 / 3.0 / 4.0 GB 显存下，**参数规模**、**比特精度**与**逐层非均匀预算分配**的翻译质量 Pareto 前沿在哪里？
* **核心对比组**：
  1. 经典基线：`FP16`（云端/高显存上限）vs `AWQ` / `GPTQ` / `GGUF Q4_K_M`
  2. 离群值旋转前沿：**SpinQuant** / **QuaRot** / **FlatQuant**（验证 W4A16 与 W4A4/KV4）
  3. 极限 2~3-bit 压榨：**QTIP** / **QuIP#** / **Unsloth Dynamic 2.0**（挑战把 7B–8B 塞进 2.5~3.2 GB）
  4. **翻译感知的逐层混合精度与弹性显存**：**EvoPress**（联合搜索逐层比特/删层配比）、**Layer-Wise Quantization / SliM-LLM**、**BitStack**（单份权重按剩余显存动态伸缩），以及剪枝-量化先后顺序消融
  5. 原生低比特与恢复：**BitNet b1.58 2B4T**、**ParetoQ** 缩放律检验、以及 **QLoRA 量化后恢复微调**

### Track 2：逐层剪枝、词表裁剪、SVD 与 MLA 改造（`experiments/tracks/02-pruning-vocab-mla/`）
* **研究问题**：除了降低每参数比特数，能否针对 **Qwen（152K 词表 + GQA + 中深层冗余）** 做结构级与逐层异构瘦身，直接腾出 20%~35% 显存？
* **核心方法**：
  1. **逐层深度剪枝与层合并（SLEB / ShortGPT / LaCo / Shortened LLaMA）**：用双语翻译校准集识别冗余中深层并迭代删层/折叠合并，配合极轻量 QLoRA 恢复。
  2. **语对定向词表裁剪（AdaptiVocab / VocabTrim）**：裁掉 Qwen 词表中非目标语种（如保留中/英/日），回收 300~600 MB Embedding & LM Head 显存并加速解码。
  3. **Qwen → DeepSeek MLA 改造（TransMLA / MHA2MLA）+ 逐层 KV 压缩（PyramidKV / KIVI 2-bit KV）**：压扁长文档/RAG 场景下的 KV Cache。
  4. **结构化剪枝 + 轻量蒸馏（Minitron + MiniLLM）** 与 **逐层异构低秩分解（SVD-LLM V2）**。
  5. **Ampere 推理加速**：评测 **Marlin / BitBLAS / EAGLE-3 + FR-Spec** 在 3060 8GB 上的真实 TTFT 与 tok/s。

### Track 3：翻译外部知识补偿（RAG / TM / 术语约束 / kNN-MT）（`experiments/tracks/03-rag-knn-mt/`）
* **研究问题**：将专有名词、领域词典与历史译文卸载出小模型权重，究竟能“等价放大”多少模型参数或抵消多少比特的量化损失？
* **核心模块**：
  1. **CPU 零显存精确术语检索**：Trie / AC 自动机（参考 WMT Efficient Terminology Integration）。
  2. **混合翻译记忆与非结构化 RAG**：CPU ONNX 多语 Embedding + BM25 + FAISS 召回相似句对（TM）与背景文档（RAGtrans）。
  3. **超越 Prompt 的解码期干预**：探索 **kNN-MT** 概率插值与轻量术语 Logit 软引导，修复低比特小模型“忽略 Prompt 术语”的退化。
  4. **质量估计路由（MetricX-QE / 困惑度）**：简单句直译，难句/低置信度句触发扩召回或短 CoT 精修。

### Track 4：OCR 与翻译全链路 8 GB 显存共存（`experiments/tracks/04-ocr-coexistence/`）
* **研究问题**：在总显存 8 GB 下，“解耦 OCR + 文本翻译小模型”与“紧凑多模态模型端到端读图翻译”各自的胜负区间？
* **核心对比路线**：
  1. **CPU 轻量化 OCR（ONNX INT8）+ GPU 3 GB Qwen + RAG**（测试 OCR 错字噪声鲁棒性与抗噪蒸馏）。
  2. **紧凑文档 OCR VLM（PaddleOCR-VL / DeepSeek-OCR）+ 2~3 GB 翻译 LLM**（显存分时/共存编排）。
  3. **端到端紧凑多模态翻译对照**（如 **TranslateGemma-4B** / **Qwen3-VL** 小尺寸，在同等总显存下做端到端精度与延迟横评）。

---

## 3. 统一科学评测指标体系

所有实验均通过 `edge-mt-lab bench` 产出统一结构的 JSON 报告，杜绝“仅凭体感或单一 BLEU 下结论”：

1. **翻译质量指标**：
   - **xCOMET / MetricX-24（含 QE）**：核心语义与细粒度错误检测指标（区分漏译、错译、风格漂移）。
   - **术语遵循率（Glossary Adherence Rate）**：给出 RAG 术语后，译文严格命中的比例。
   - **格式与占位符保真率**：Markdown / 表格 / HTML / 变量占位符保留率。
   - **OCR 噪声鲁棒性（Noise Degradation）**：在模拟/真实 OCR 错字样本上的质量掉点幅度。
2. **硬件效率指标**：
   - **全链路峰值显存（Peak VRAM MB）**：包含 CUDA Context + 模型权重 + KV Cache +（若在 GPU）OCR 显存。
   - **首字延迟 TTFT (ms)** 与 **解码吞吐 Decode (tok/s)**。
   - **Pareto 效率曲线**：以 **VRAM (2 / 3 / 4 GB)** 为横轴，**xCOMET / 术语遵循率** 为纵轴绘制前沿包络线。
