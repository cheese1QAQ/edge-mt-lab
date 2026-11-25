# Track 3：翻译外部知识补偿（RAG / TM / 术语约束 / kNN-MT）

## 核心实验问题
定量测量外挂非参数知识系统对 2~4 GB 轻量化小模型的“能力杠杆”：

- **零显存 CPU 精确术语检索**：Trie / AC 自动机抽取命中术语（< 1ms，0 VRAM）。
- **混合翻译记忆与非结构化文档 RAG（TM-RAG / RAGtrans）**：CPU ONNX 多语 Embedding + BM25 + FAISS。
- **解码期术语干预（kNN-MT / Logit Bias）**：解决低比特小模型忽略 Prompt 术语约束的问题。
- **无参考质量估计路由（MetricX-QE / 困惑度）**：低置信度长难句自动触发二次检索或短 CoT 精修。
