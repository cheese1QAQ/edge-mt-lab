# 实验目录与报告规范

本目录按四大研究主线组织实验脚本、配置与结构化评测报告：

- [Track 1：前沿权重量化与比特–规模缩放律](tracks/01-quantization/README.md)
- [Track 2：结构化剪枝、词表裁剪、SVD 与 MLA 改造](tracks/02-pruning-vocab-mla/README.md)
- [Track 3：翻译外部记忆补偿（RAG / TM / 术语 / kNN-MT）](tracks/03-rag-knn-mt/README.md)
- [Track 4：8 GB 显存下的 OCR 与翻译全链路共存](tracks/04-ocr-coexistence/README.md)
- `reports/`：存放由 `edge-mt-lab bench` 导出的结构化 JSON 实验报告（命名规范：`<date>_<track>_<model>_<method>.json`）。

> [!CAUTION]
> **切勿将 GB 级模型权重（`.gguf`、`.safetensors`、`.onnx`、`.bin`）或原始大型语料提交进 Git。** 仅提交可复现配置、脚本与轻量 JSON 报告。
