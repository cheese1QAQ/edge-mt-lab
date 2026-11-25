# 评测与校准数据集说明

## `eval_sample.jsonl`

`eval_sample.jsonl` 是一个**小规模、零版权风险、可在 GitHub Actions CI 中干跑校验**的多语种翻译冒烟测试集，覆盖四个核心评测维度：

1. `terminology`：含专有名词/技术术语约束（测试术语遵循率与 RAG 注入）
2. `ocr_noise`：含典型 OCR 识别噪声（如形近字混淆、换行断句、标点错识），测试模型抗噪与纠错翻译能力
3. `formatting`：含 Markdown / 代码占位符 / 表格标记，测试低比特量化后的格式保真率
4. `discourse`：含跨句指代与上下文背景提示，测试文档级一致性与长上下文 KV 压缩后的保真度

### 字段 Schema

每行一个 JSON 对象，由 `edge_mt_lab.schemas.EvalSample` 严格校验：
- `id` (`str`)：唯一样本编号
- `source_lang` / `target_lang` (`str`)：BCP-47 语言码（如 `en`, `ja`, `zh-CN`）
- `category` (`terminology | ocr_noise | formatting | discourse`)：评测维度
- `source` (`str`)：待翻译原文（可能含 OCR 噪声或占位符）
- `reference` (`str`)：参考译文
- `glossary` (`dict[str, str]`)：外部 RAG / Trie 检索命中的强制术语映射
- `context_notes` (`str`)：可选背景摘要或翻译记忆提示
- `ocr_corrupted` (`bool`)：是否属于 OCR 噪声样本
