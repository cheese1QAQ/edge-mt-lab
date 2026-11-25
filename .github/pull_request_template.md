## 变更摘要

<!-- 简述本次 PR 新增的文献笔记、实验代码、评测报告或架构改进 -->

## 关联研究主线 / 议题

- Track: <!-- Track 1 (Quantization) / Track 2 (Pruning-Vocab-MLA) / Track 3 (RAG-kNN-MT) / Track 4 (OCR Coexistence) -->
- Closes #

## 变更类型

- [ ] 论文精读笔记 (`notes`)
- [ ] 轻量化 / 翻译 / OCR 实验与评测报告 (`research`)
- [ ] 工具链或评测框架功能 (`feat`)
- [ ] 缺陷修复 (`fix`)
- [ ] 文档更新 (`docs`)
- [ ] 工程 / CI 维护 (`chore` / `ci`)

## 验证检查清单

- [ ] 本地运行 `make check` 通过（`ruff` + `mypy --strict` + `pytest`）
- [ ] 本地运行 `make doctor` 与 `make bench-dry` 通过
- [ ] 未提交模型权重文件（`.gguf` / `.safetensors` / `.onnx` / `.pt`）或大体积本地缓存
- [ ] 如为实验报告，已在报告中完整记录硬件显存峰值（含 KV Cache 与 OCR 占用）与复现命令
