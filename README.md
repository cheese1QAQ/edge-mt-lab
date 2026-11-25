# Edge-MT-Lab — 2~4 GB 显存预算下的前沿 LLM 轻量化与翻译系统研究

[![CI](https://img.shields.io/badge/CI-GitHub%20Actions-2088FF?logo=githubactions&logoColor=white)](.github/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Edge-MT-Lab** 是一个独立的开源研究实验室，聚焦**前沿大语言模型轻量化与推理系统优化**，以 **Qwen** 与 **DeepSeek** 生态为双主线基座，探索如何在严格受限的消费级硬件（以 **RTX 3060 8GB / Ampere SM86** 为基准，**LLM 显存预算严控在 2~4 GB**，并预留显存与算力给 **OCR / 视觉模块**）上，通过 **“极限模型压缩 × 底层推理加速 × RAG/外部记忆能力补偿”** 实现高可用机器翻译。

---

## 核心研究命题

1. **比特–规模 Pareto 前沿**：在严格 **2 / 3 / 4 GB 显存预算**内，同样体积下“8B @ 2~3-bit（QTIP / SpinQuant / Dynamic 2.0）”与“4B @ 4-bit”以及“8B 结构化剪枝蒸馏→4B（Minitron）”与“原生 1.58-bit（BitNet）”谁在翻译任务上最优？
2. **Qwen × DeepSeek 架构级融合瘦身**：将 **语对词表裁剪（AdaptiVocab / VocabTrim）**（削减 Qwen 152K 词表冗余）与 **TransMLA / MHA2MLA**（将 GQA 转为 DeepSeek 风格低秩 MLA）及 **KIVI 2-bit KV** 叠加，量化评估显存节省与长文翻译质量损益。
3. **非参数外部记忆的能力杠杆**：通过 **CPU 零显存 Trie 术语表 + 翻译记忆 TM-RAG + 解码期术语软约束 / kNN-MT + QE 路由精修**，把专有名词和领域知识卸载出模型权重，定量回答“外部系统能抵消多少比特的量化退化”。
4. **8 GB 卡上的 OCR + 翻译全链路共存**：对比「CPU 轻量化 OCR + GPU 3 GB Qwen + RAG」与「紧凑 VLM / DeepSeek-OCR / PaddleOCR-VL 管线」在含噪声截图与文档翻译上的真实延迟、峰值显存与准确率。

---

## 仓库目录结构

```text
edge-mt-lab/
├── .github/                  # GitHub Actions CI/Release、Dependabot、Issue & PR 模板
├── docs/
│   ├── research-plan.md      # 硬件物理边界分析、四大研究主线与评测体系
│   └── reading-list.md       # 62 篇经核验的前沿论文精读清单（Qwen/DeepSeek 路线）
├── notes/                    # 论文精读笔记库与统一笔记模板
├── experiments/
│   ├── tracks/               # 4 条研究主线实验目录与记录规范
│   └── reports/              # 结构化 JSON 评测报告（Pareto 曲线数据源）
├── datasets/                 # 评测集 Schema 与小规模多语种/OCR 噪声冒烟测试集
├── src/edge_mt_lab/          # Python 研究工具链：硬件预算估算器、Schema 与 Benchmark 套件
├── tests/                    # 零 GPU 可运行的单元测试（CI 门禁）
├── Makefile                  # 统一开发、检查与实验命令入口
└── CONTRIBUTING.md           # GitHub Flow、Conventional Commits 与可复现实验规范
```

---

## 快速开始

```bash
# 1. 使用 uv 安装开发环境与质量工具链
make sync

# 2. 运行完整质量门禁（ruff lint/format + mypy --strict + pytest）
make check

# 3. 估算不同 Qwen / DeepSeek 压缩方案在 RTX 3060 8GB (2~4GB LLM 预算) 下的显存拆分
make doctor

# 4. 校验评测集 Schema 并执行基准框架干跑（零 GPU）
make bench-dry
```

---

## 导航文档

- [研究总方案与四大实验主线](docs/research-plan.md)
- [前沿文献精读清单（62 篇核验论文 + 四周路线图）](docs/reading-list.md)
- [论文笔记索引与模板](notes/README.md)
- [实验目录与报告规范](experiments/README.md)
- [贡献与可复现规范](CONTRIBUTING.md)
