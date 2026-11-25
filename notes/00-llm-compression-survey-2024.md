# [LLM Compression Survey] A Survey on Model Compression for Large Language Models

- **年份 / 会议**：2024 / TACL (*Transactions of the Association for Computational Linguistics*, Vol. 12, pp. 1556–1577)
- **作者**：Xunyu Zhu, Jian Li (通讯作者), Yong Liu, Can Ma, Weiping Wang（中科院信工所 / 国科大 / 人大高瓴 / 北师大）
- **链接**：[arXiv:2308.07633](https://arxiv.org/abs/2308.07633)（v4: 2024-07-30）｜ [TACL 正式版](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00704/125482/A-Survey-on-Model-Compression-for-Large-Language)
- **交互式思维导图**：[00-llm-compression-survey-2024-mindmap.html](00-llm-compression-survey-2024-mindmap.html)（本地离线可直接用浏览器打开，含 17 大核心盲点扫盲卡片）
- **对应研究主线**：模块 0（全景奠基，串联 Track 1 / Track 2 / Track 3 / Track 4）

---

## 1. 解决的核心瓶颈

大模型参数规模与推理/部署资源需求巨大（例如 GPT-175B 仅 FP16 权重即占 **350 GB** 显存，至少需 5 张 80GB A100 GPU 才能部署推理）。作为**首篇专门面向 LLM 的模型压缩系统综述**，本文系统梳理了四大核心压缩范式（量化、剪枝、知识蒸馏、低秩分解）、效率与能力评测体系、7 条关键工程/科研洞见（Remarks 1–7）以及 6 大未来开放挑战，并明确区分：

- **显存墙**：静态权重显存、长上下文 KV Cache 显存、训练反向传播显存。
- **速度墙**：Decode 阶段卡显存带宽（Memory-bound）vs. Prefill/大 Batch 阶段卡峰值算力（Compute-bound），强调理论 `FLOPs` 降低不等于真实提速，必须关注 **MFU** 与实测 **Speedup Ratio**。

---

## 2. 核心方法与数学直觉（逐章精读梳理）

### 一句话总结
按**任务相关性（Figure 2）**，模型压缩分为**任务相关型**（QAT、KD，需要再训练/微调）与**任务无关型**（PTQ、剪枝、低秩分解，免任务微调直接作用于预训练权重，但 PTQ 与剪枝高度依赖校准集质量）。

### 2.1 评测指标与基准（Section 2）
- **六大效率指标**（须与准确率、零样本能力联合报告）：
  - `Model Size`（总参数量/显存占用）、`FLOPs`（理论浮点运算总量）、`MFU`（平均算力利用率 = 实际 FLOPS / 硬件理论峰值 FLOPS）、`Inference Time`（端到端延迟）、`Speedup Ratio`（未压缩延迟 / 压缩后延迟）、`Compression Ratio`（原始体积 / 压缩后体积）。
- **四大评测体系**：
  1. 经典基准：PPL（WikiText-2 / C4 / PTB）、Zero-shot（LAMBADA / PIQA / OpenBookQA）、Reasoning（GSM8K / CommonsenseQA / StrategyQA）。
  2. 综合高难基准：BIG-Bench（200+ 任务，其高难子集为 BBH）。
  3. 未见指令数据集：Vicuna-Instructions（80 题）、User-Oriented-Instructions（252 题）。
  4. 统一复现框架：EleutherAI LM Evaluation Harness（60+ 标准基准）。

### 2.2 量化（Quantization, Section 3）
- **QAT（量化感知训练，Sec 3.1）**：边量化边重训，主攻 1–2 bit 极限：`LLM-QAT`（自生成数据免外部语料蒸馏，W4A8）、`BitDistiller`（QAT + 自蒸馏 + Confidence-Aware KL，主攻 2-bit）、`OneBit`（1-bit 参数表示与初始化）。
- **PTQ（训练后量化，Sec 3.2）三大分支**：
  - **仅权重量化（Weight-Only，3–4 bit 甜点区）**：靠大幅降低显存搬运量在访存受限的 Decode 阶段实现 2–3× 加速（需片上反量化，不用低比特整数 Tensor Core）。
    - *误差补偿/矩阵变换派*：`LUT-GEMM`（BCQ 查找表）、`GPTQ`（逐层逆 Hessian 误差补偿）、`QuIP`（Hessian LDL + 随机正交矩阵非相干处理，推至 2-bit）。
    - *敏感权重保护派*：`AWQ`（按激活幅值识别 Top 1% 敏感权重通道；注：原论文靠逐通道缩放等效保护而非真正混存 FP16）、`OWQ`、`SpQR`（L2 预测误差 + 稀疏高精存储）、`SqueezeLLM`（Hessian 加权 k-means 非均匀量化 + 稀疏高精异常值）。
  - **权重-激活联合量化（W-A）**：解锁硬件低比特算力，核心痛点是驯服**激活异常值（Activation Outliers）**。
    - *基线*：`ZeroQuant`（分组权重 + 按 token 激活，INT8）；*隔离派*：`LLM.int8()`（异常维走 FP16，W8A8 无损）；*等效迁移派*：`SmoothQuant`（逐通道缩放 $Y = (X \cdot \text{diag}(s)^{-1})(\text{diag}(s) \cdot W)$ 将难度从激活转嫁给权重）、`OS+`（逐通道平移 + 缩放，W4A4）、`OmniQuant`（可学习等效变换 + 可学习权重截断，W4A6）；*重排与格式派*：`RPTQ`（通道聚类重排）、`OliVe`（OVP 异常值-牺牲品配对，W4A4 加速 4.5×）、`LLM-FP4`（浮点指数位覆盖大动态范围）。
  - **KV Cache 量化**：攻克长上下文显存墙。`KVQuant`（Key 按通道 + Pre-RoPE 量化 + 非均匀量化，支持千万级上下文）、`KIVI`（揭示核心规律：**Key 按通道、Value 按 token**，免微调 2-bit）、`WKVQuant`（权重 + KV 联合 4-bit，W4A16KV4）。

### 2.3 剪枝（Pruning, Section 4）
核心矛盾：**精度保留 vs. 硬件真实加速（MFU）**。
- **非结构化剪枝（Sec 4.1）**：删单个权重，免重训即可 50% 稀疏几乎无损，但通用 GPU 难提速。代表：`SparseGPT`（逐层逆 Hessian 补锅）、`Wanda`（$|W_{ij}| \cdot \|X_j\|_2$，免 Hessian/免权重更新/免重训，呼应激活异常值直觉）、`SAMSP`（按 Hessian 跨层动态分配稀疏率）、`DSnoT`（迭代剪枝-再生长，但 90% 稀疏仍崩盘）、`Flash-LLM`（全局显存稀疏存省带宽 + 片上 SRAM 还原稠密块送 Tensor Core）。
- **结构化剪枝（Sec 4.2）**：删整组通道/注意力头（宽度）或整层 Block（深度），矩阵真实变小、硬件无关提速，但掉点较明显，常需 LoRA 或蒸馏恢复。
  - *Loss/梯度派*：`LLM-Pruner`（依赖检测 + 剪宽度 + LoRA）、`Shortened LLaMA`（剪深度 Block + LoRA）。
  - *幅值/信号派*：`FLAP`（波动性指标 + 偏置补偿，免微调）、`SliceGPT`（利用计算不变性做逐层 PCA 旋转后直接切片，30% 剪枝提速 1.87× 且 ΔPPL 仅 +1.73）。
  - *正则化派*：`Sheared LLaMA`（拉格朗日乘子约束目标形状 + 动态批加载持续预训练）。
- **半结构化剪枝（Sec 4.3，N:M 稀疏）**：每 $M$ 个连续权重保留 $N$ 个非零值。`E-Sparse`（信息熵 + 通道重排）、`SparseGPT (2:4)`、`Wanda (2:4)`。NVIDIA Ampere Tensor Core 原生加速 **2:4**，但仅限 2:4 比例。

### 2.4 知识蒸馏（Knowledge Distillation, Section 5）
直接让大教师（Teacher）训练独立小模型（Student）：
- **黑盒蒸馏（Sec 5.1，只访问教师文本输出）**：重点迁移三大涌现能力：
  - *思维链（CoT）*：多任务解释学习（Distilling Step-by-Step）、双学生拆解+求解、代码思维 `PaD`（Program-of-Thought）、防捷径反事实目标、交互自我反思 `DRA`、以及从**负样本错题**学习；警示（Fu et al.）：小模型强行专精单科常损害通用能力。
  - *上下文学习（ICL）*：`Meta-ICT` / `Multitask-ICT` / `AICD`。
  - *指令遵循（IF）*：`Lion`（造困难指令）、`LaMini-LM`（258 万指令）、`SELF-INSTRUCT`（自举）、`Selective Reflection-Tuning`（学生按兼容性择优吸收）。
- **白盒蒸馏（Sec 5.2，访问 logits 分布与中间隐层）**：
  - `MINILLM`：改用 **反向 KL 散度（Reverse KL）** 聚焦教师分布主峰（mode-seeking），避免小模型因强行覆盖长尾低概率区而产生幻觉。
  - `GKD`（On-Policy 蒸馏）：让学生在**自己生成的序列**上接受教师实时反馈，缓解训练-推理分布偏移（exposure bias）。
  - `TED`：任务感知滤波器逐层对齐中间隐藏状态。

### 2.5 低秩分解（Low-Rank Factorization, Section 6）
将 $W_{m \times n} \approx U_{m \times k} V_{k \times n}$（$k \ll m, n$），同时压参数和 FLOPs 且产物仍为**标准稠密矩阵**（免稀疏硬件算子）：
- `LPLR`（随机素描低秩 + 低精度双压缩）；
- `ASVD`（激活感知 SVD：按输入激活通道分布缩放权重后再 SVD + 异构逐层压缩率）；
- `LASER`（层级选择性截断特定层高阶奇异分量，反而起到去噪效果，显著提升罕见事实回忆与抗问句改写能力）。

### 2.6 作者提炼的 7 条核心洞见（Remarks 1–7）
1. **Remark 1（QAT + PEFT）**：全参 QAT 太贵，结合 PEFT（从任务相关的 QLoRA/PEQA/LoftQ 到更通用的 L4Q）极具潜力。
2. **Remark 2（结构化剪枝 + KD）**：用知识蒸馏恢复结构化剪枝损失是黄金组合（如后续 NVIDIA Minitron）。
3. **Remark 3（动态剪枝）**：按当前任务/输入动态激活不同子网络，更好平衡多任务性能与效率。
4. **Remark 4（校准集至关重要）**：PTQ 和剪枝对校准集分布极其敏感，选错校准集会导致下游任务剧烈退化。
5. **Remark 5（黑盒蒸馏的泛化瓶颈）**：依赖有限文本样本难以覆盖 OOD 输入，数据多样性亟待提升。
6. **Remark 6（黑盒 + 白盒两段接力）**：先「闭源强模型 → 黑盒蒸馏 → 开源教师」，再「开源教师 → 白盒蒸馏 → 端侧小模型」。
7. **Remark 7（挖掘内部结构）**：白盒蒸馏应更深入利用 Transformer 层间连接与内部交互机制。

### 2.7 六大挑战与未来研究方向（Section 7）
1. **更先进的压缩方法（7.1）**：缩小压缩模型与未压缩原模型的精度鸿沟。
2. **经典小模型压缩术的大尺度化（7.2）**：将彩票假设、参数共享等扩展到百亿/千亿 LLM。
3. **面向真实硬件的推理与部署（7.3）**：引入 **Roofline Model（屋顶线模型）** 综合评估算术强度、显存与吞吐。
4. **Scaling Law 的制约（7.4）**：建立压缩态下的 Scaling Law 理论。
5. **AutoML 赋能 LLM 压缩（7.5）**：用 Meta-Learning / NAS 自动搜索学生架构与逐层压缩配比。
6. **压缩的可解释性（7.6）**：揭示 CoT 蒸馏机理及压缩保留/擦除知识的规律。

---

## 3. 前置依赖与硬件约束

- **是否需要重训 / 校准集 / 额外数据**：
  - `PTQ`（AWQ/GPTQ/SmoothQuant/KIVI）、`非结构化剪枝`（SparseGPT/Wanda）、`FLAP/SliceGPT`、`ASVD` 均**免重训**，仅需少量高质量校准集（通常 128–512 条样本），**Google Colab 单卡几分钟即可跑完**。
  - `QAT`、`结构化剪枝恢复`（LLM-Pruner/Shortened LLaMA）、`KD` 需要训练/微调数据与反向传播（Colab 单卡优先采用 QLoRA / 离线黑盒蒸馏 / Top-K logits 轻量化白盒蒸馏）。
- **对 RTX 3060 8GB (Ampere SM86) 的兼容性**：
  - **W4A16 仅权重量化（AWQ / GPTQ）+ KV Cache 量化（KIVI / KV4）**：最契合 3060 240 GB/s 访存受限解码特性，已有成熟的 Marlin / `llama.cpp` / vLLM 算子支持。
  - **2:4 半结构化稀疏**：Ampere SM86 硬件原生支持 2:4 Tensor Core，但严格锁死 2:4 比例且低 batch 解码下显存收益弱于 4-bit 量化。
  - **FP8 / FP4 原生硬件格式**：Ampere 不原生支持 Hopper FP8 / Blackwell FP4，需靠反量化或软件/BitBLAS 算子。

---

## 4. 关键实验结论（论文 Table 1 & Table 2 精粹）

> **精读防坑提示**：Table 1 与 Table 2 所有数字均直接摘自各方法原论文，不同行基座模型规模差异巨大（1.3B–175B），且大模型通常比小模型更抗压缩，**不可直接跨行比“冠军”**，应重点观察同档位趋势。

| 范式 / 代表方法 | 基座模型 | 配置 | 质量变化（WikiText-2 ΔPPL） | 实测加速比 |
| :--- | :--- | :--- | :--- | :--- |
| **QAT** · LLM-QAT / BitDistiller / OneBit | LLaMA-30B / 2-13B / 13B | W4A8 / W2 / W1 | +0.5 / +1.9 / +4.09 | - |
| **Weight-Only PTQ** · GPTQ / AWQ / SpQR | OPT-175B / LLaMA2-70B / 30B | W3 / W3 / W3.89 | +0.34 / +0.42 / +0.15 | **3.24× / 3.2× / 2.0×** |
| **Weight-Only 2-bit** · QuIP | LLaMA2-70B | W2A16 | +3.007 | - |
| **W-A PTQ (8-bit)** · LLM.int8() / SmoothQuant | OPT-13B / OPT-175B | W8A8 | **0.00 / +0.18** | 1.22× / 1.56× |
| **W-A PTQ (4-bit)** · RPTQ / OliVe / OS+ | OPT-175B / BLOOM-7B / LLaMA-65B | W4A4 | +2.26 / +2.11 / +5.77 | - / **4.5×** / - |
| **KV Cache PTQ** · KVQuant / WKVQuant | LLaMA-65B / LLaMA-13B | KV2 / W4KV4 | **+0.19 / +0.12** | 1.4× / - |
| **非结构化剪枝** · SparseGPT / Wanda / DSnoT | OPT-175B / LLaMA-65B | 50% / 50% / 90% | **-0.14** / +1.01 / `2.08e4`(崩盘) | `-`（难直接加速） |
| **结构化剪枝** · LLM-Pruner / SliceGPT / FLAP | LLaMA-13B / 2-70B / 65B | 20% / 30% / 50% | +3.6 / **+1.73** / +7.09 | - / **1.87×** / - |
| **2:4 半结构化** · SparseGPT / E-Sparse / Wanda | OPT-175B / LLaMA-65B | 2:4 (50%) | **+0.39** / +2.13 / +2.69 | **2× / 1.53× / 1.24×** |

---

## 5. 可迁移到 Edge-MT-Lab 的实验假设

- [x] **可直接作为对比基线与框架指南**：将 `AWQ / GPTQ (W4A16)` + `KIVI (KV4/KV2)` 作为 2–4 GB 预算下的第一批强基线。
- [x] **可与其他方法正交组合**：
  - 遵循 **Remark 4**：在翻译任务上设立「**通用维基校准集 vs. 翻译领域校准集**」消融实验，验证校准集对 3–4 bit 量化与剪枝的真实影响。
  - 遵循 **Remark 2 + Sec 5.2**：用开源强教师结合 `MINILLM` 反向 KL 思想修复结构化剪枝或蒸馏 1.5B–4B 翻译小模型。
  - 遵循 **Sec 6 + Sec 3**：验证「低秩分解（ASVD/SVD-LLM）+ 4-bit 量化」或「模块替换（如 TransMLA / DeepEncoder）→ FP16 对齐 → 统一量化」的多维正交叠加。
- [x] **潜在风险 / 失败模式**：
  - 仅看 WikiText-2 PPL 容易掩盖低比特下**术语错译、漏译与长上下文退化**（需配合 xCOMET / 术语命中率评测）。
  - 综述截至 2024 年中，需衔接其后的旋转量化（QuaRot/SpinQuant）、2-bit 格点/网格量化（QuIP#/QTIP）、1.58-bit 缩放律（ParetoQ）与词表裁剪（AdaptiVocab）。

---

## 6. 延伸问答与思考记录（精读讨论 Q1–Q6）

### Q1：除了量化、剪枝、蒸馏、低秩分解，能否通过替换更先进的模块（如 DeepSeek 视觉减 Token + 坐标定位方法）提效？
- **结论**：完全可以，且与四大参数压缩技术**正交可叠加**。广义轻量化包含四层：**参数压缩层**（本论文）× **输入/Token 压缩层**（直接削减 $O(L^2)$ 注意力）× **高效架构模块替换层**（MoE、GQA/MLA、线性注意力）× **推理系统优化层**（FlashAttention、PagedAttention、投机解码）。
- **案例剖析（DeepSeek-OCR, `arXiv:2510.18234`）**：其 `DeepEncoder` 采用「SAM-base 局部窗口注意力 → 2 层卷积 16× 下采样 → CLIP-large 全局注意力」，在保留规则 2D 网格空间拓扑与 grounding 坐标监督的同时将视觉 Token 大幅压减（7–20×），再配合 `DeepSeek-3B-MoE-A570M` 解码器实现多维提效。

### Q2：直接替换模块是否需要重训？应该在量化前还是量化后做？
- **结论**：除数学等价算子替换（如 FlashAttention）外，改变特征空间或结构的模块替换**需要局部对齐微调或继续预训练**；且压倒性推荐 **先在未量化高精度（FP16/BF16）模型上替换并训稳，最后再统一量化**。
- **原因**：① 离散低比特权重含噪声不利于梯度稳定更新；② 替换模块会改变下游激活分布，导致旧 PTQ 缩放/截断参数失效；③ 最后统一量化可按模块敏感度差异化设位宽（如视觉编码器保 FP16/FP8、LLM 主干压 INT4）。若训练显存不足，可用 **QLoRA**（冻结 4-bit 主干 + FP16 训新模块与 LoRA），训完合并回 FP16 后再统一校准量化。

### Q3 & Q4：无需反向传播（BP-free）的训练方法（确认为 EGGROLL，`arXiv:2511.16652`）用于 LLM 代价是否过大？算力有限（Colab）能否使用？
- **EGGROLL 核心机制**：将进化策略（ES）各候选个体的扰动矩阵结构化为**低秩矩阵** $E_i = A_i B_i^\top$（大模型上甚至 $r=1$ 即有效），让所有候选个体共享主权重做高算术强度批量前向，吞吐接近纯批量推理的 69–91%，而所有个体加权求和后的整体更新仍可达满秩 $\min(Nr, m, n)$。
- **Colab / 有限算力避坑要点**：
  1. **绝不能用于通用从头预训练**：论文中 int8 RNN 预训练胜过 BP 依赖百万级种群，总 GPU-hours 约为 BP 的 **180 倍**。
  2. **标准 Transformer 受 KV Cache 制约**：长序列下 KV Cache 会限制单卡最大并发种群规模（论文主力微调选了固定隐状态的 RWKV-7）。
  3. **选型边界**：常规可微 SFT / 蒸馏在 Colab 上仍首选 **QLoRA + 梯度检查点 + FlashAttention**；而在**不可微结果奖励后训练（对标 GRPO，如数学对错、代码通过率、翻译 BLEU/COMET）**或**直接在低比特/纯整数不可微权重上微调**时，EGGROLL 才具有真正优势。

### Q5：知识盲点较多、难以串联怎么办？
- 按 **「病因诊断（显存三大件 & 带宽墙/算力墙）→ 评测照妖镜（FLOPs/MFU/PPL）→ 广义轻量化 4 层 → 量化/剪枝/蒸馏/低秩与未来方向」** 的因果链条梳理为交互式离线思维导图 [00-llm-compression-survey-2024-mindmap.html](00-llm-compression-survey-2024-mindmap.html)，提炼 **17 大高频核心盲点** 逐一配通俗类比与可勾选进度追踪。

### Q6：在垂直剧本翻译（如美少女游戏 / Galgame）场景下，能否结合「少量前人优质人工翻译 + 强黑盒模型合成对照语料」训练小模型？
- **技术评价**：可行，本质为机器翻译经典的 **序列级知识蒸馏（Seq-KD）+ 人工精修**。
- **合规与平替**：主流闭源商业 API 的 ToS 通常禁止蒸馏训练模型；2025–2026 年建议改用许可宽松的**强开源权重教师**（如 MIT/Apache 2.0 协议的 DeepSeek-V3/R1、Qwen3-32B/235B），合法且成本更低。商业游戏文本与汉化语料受版权保护，切勿公开分发原语料，评测集须按**不同作品严格切分**防泄漏。
- **数据工程 4 条要点**：① 必须带上下文窗口 + 说话人 + 人设/术语表生成（避免日语省略主语与口癖丢失）；② 两阶段用法（合成数据 QLoRA 打底 → 稀缺人工精翻精修，并配对构造「人工精翻 vs 僵硬机翻」做轻量 CPO/DPO 去翻译腔）；③ 用无参考 QE（MetricX-24 QE / xCOMET-QE）+ 规则筛除噪声合成句；④ 在训练 Prompt 中显式训练小模型“严格遵守外部传入术语表”的泛化指令能力。
