# Track 2：结构化剪枝、词表裁剪、SVD 与 MLA 改造

## 核心实验问题
针对 **Qwen（152K 词表 + GQA）** 与 **DeepSeek（MLA + MTP）** 的架构特征，验证结构级轻量化叠加量化后的真实收益：

- **语对词表裁剪（AdaptiVocab / VocabTrim）**：仅保留目标语对（如中/英/日）高频词表，实测 Embedding + LM Head 显存回收与 LM Head 解码提速。
- **GQA → MLA 改造（TransMLA / MHA2MLA）+ KIVI 2-bit KV**：压低长文档/多轮 RAG 上下文的 KV Cache 占用。
- **结构化剪枝 + 蒸馏（Minitron + MiniLLM）** 与 **SVD-LLM V2** 低秩压缩。
- **Ampere 底层推理加速**：Marlin / BitBLAS / EAGLE-3 + FR-Spec 在 RTX 3060 8GB 上的带宽利用率对比。
