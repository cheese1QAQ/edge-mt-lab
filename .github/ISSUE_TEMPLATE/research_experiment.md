---
name: Research Experiment
about: 登记一项 2~4GB 显存预算下的 LLM 轻量化、RAG 翻译补偿或 OCR 共存实验
title: "research: "
labels: ["research"]
---

### 研究假设与目标

### 硬件与显存约束
- 目标显卡: <!-- 例如 RTX 3060 8GB (Ampere SM86, 240 GB/s) -->
- LLM 总显存上限: <!-- 2 GB / 3 GB / 4 GB -->
- 共存模块（OCR / Embedding / 检索）: <!-- CPU ONNX / GPU 轻量化 VLM -->

### 实验矩阵
- 基座模型（Qwen / DeepSeek 系列）:
- 压缩方法（如 SpinQuant / QTIP / Dynamic 2.0 / Minitron / AdaptiVocab / TransMLA / KIVI）:
- 外部补偿机制（如 Trie 术语表 / TM-RAG / kNN-MT / QE 路由）:
- 校准集与评测集:

### 核心评测指标
- 峰值显存（含 CUDA Context + 权重 + KV Cache）、TTFT (ms)、Decode (tok/s)
- xCOMET / MetricX-QE、术语遵循率、OCR 噪声鲁棒性
