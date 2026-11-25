# 贡献与实验协作指南（GitHub 仓库维护规范）

本项目参考现代开源研究与工程规范，确保每一项轻量化实验、文献笔记与基准数据都具备可复现性。

## 1. 分支模型（GitHub Flow）

- `main` 为主干分支，始终保持 CI 全绿、文档链接有效、基准干跑通过。
- 所有研发与实验记录均从 `main` 切出短生命周期分支并通过 Pull Request 合并：
  - `research/<topic>`：量化、剪枝、词表裁剪、MLA、RAG-MT、OCR 协同实验与报告（例如 `research/qwen3-4b-spinquant-vs-awq`）
  - `notes/<paper>`：论文精读笔记（例如 `notes/paretoq-2025`）
  - `feat/<topic>`：评测脚本、硬件预算工具、RAG/解码实验代码
  - `fix/<topic>`：缺陷修复
  - `docs/<topic>`：文档、路线图与文献清单更新
  - `chore/<topic>`：工程配置、依赖与 CI 维护

## 2. 提交规范（Conventional Commits）

采用 `<type>(<scope>): <subject>` 格式，便于自动生成 Changelog 与 Release Notes：

- `research(quant): benchmark Qwen3-4B AWQ vs SpinQuant under 3GB VRAM`
- `notes(mla): add reading notes for TransMLA and MHA2MLA`
- `feat(bench): add xCOMET and OCR-noise robustness metrics`
- `fix(hardware): fix Ampere SM86 KV cache estimation formula`
- `ci(actions): enable Python 3.10 and 3.12 test matrix`

## 3. 提交 PR 前本地检查清单

```bash
make check      # ruff lint + format check + mypy --strict + pytest
make doctor     # 校验硬件与 2~4GB 显存预算计算器
make bench-dry  # 校验评测数据集 Schema 与干跑基准管线
```

## 4. 研究实验可复现规范

向 `experiments/` 提交实验结果时，必须同时保留：
1. **确切基座与压缩配置**：模型版本（如 `Qwen/Qwen3-4B`）、量化/剪枝/词表裁剪参数、校准集来源。
2. **硬件与运行时边界**：GPU 型号（如 RTX 3060 8GB）、CUDA/引擎版本（`llama.cpp` / vLLM / BitBLAS / Marlin）、KV Cache 精度、是否同时驻留 OCR 模块。
3. **结构化评测报告**：由 `edge-mt-lab bench` 导出的 JSON 报告放入 `experiments/reports/`。**严禁将任何模型权重（`.gguf` / `.safetensors` / `.onnx` / `.bin`）提交到 Git。**
