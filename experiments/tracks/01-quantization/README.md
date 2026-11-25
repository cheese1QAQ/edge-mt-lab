# Track 1：前沿权重量化与比特–规模缩放律

## 核心实验问题
在严格 **2.0 GB / 3.0 GB / 4.0 GB** LLM 总显存预算下，对比 Qwen / DeepSeek 生态模型在以下方法中的翻译质量与 3060 8GB 实测吞吐：

- **Baseline**：FP16（上限对照）、AWQ (W4A16)、GGUF `Q4_K_M` / `IQ4_XS`
- **旋转消离群值**：QuaRot、SpinQuant、FlatQuant（W4A16 与 W4A4/KV4）
- **极限 2~3-bit**：QTIP、QuIP#、Unsloth Dynamic 2.0（重点验证 `Qwen3-8B @ 2.5-bit` vs `Qwen3-4B @ 4-bit`）
- **逐层混合精度与动态显存**：EvoPress、Layer-Wise Quantization、SliM-LLM、BitStack（以及剪枝-量化先后顺序消融）
- **原生 1.58-bit & 恢复微调**：BitNet b1.58 2B4T、ParetoQ 缩放律验证、QLoRA 量化后领域恢复
