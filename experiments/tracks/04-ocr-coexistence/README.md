# Track 4：8 GB 显存下的 OCR 与翻译全链路共存

## 核心实验问题
在 **RTX 3060 8GB** 整卡约束下，系统评测视觉文字提取与机器翻译的跨模块显存编排与抗噪表现：

- **路线 A（极致显存隔离）**：CPU ONNX 轻量 OCR + GPU 2~4 GB Qwen 翻译模型 + RAG（附 OCR 错字噪声鲁棒性评测）。
- **路线 B（双轻量 GPU 模型编排）**：紧凑文档 OCR VLM（PaddleOCR-VL ~0.9B / DeepSeek-OCR）与 2~3 GB 翻译 LLM 共存或分时切换。
- **路线 C（端到端紧凑多模态对照）**：TranslateGemma-4B / Qwen3-VL 小尺寸在同等总显存下的端到端读图翻译对比。
