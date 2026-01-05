# 12. 去重优先级明确（Deduplication Priority）

## 问题背景

实体去重时，需要明确 name、summary 和 attributes（如 code）的优先级关系。

**错误做法**：让 LLM 过度依赖 code 属性，导致可能将 code 相同但实际不同的实体错误合并。

**正确做法**：name 和 summary 是主要判断依据，attributes 是辅助证据。

## 解决方案

在 dedupe prompt 中明确去重优先级：

**文件**: `graphiti_core/prompts/dedupe_nodes.py`

在 `node()` 和 `nodes()` 函数中添加：

```python
**Deduplication Priority** (in order of importance):
1. **Name + Summary**: Primary evidence. If name and summary clearly describe the same real-world concept, they are duplicates.
2. **Attributes as supporting evidence**: When names differ but summaries suggest the same concept, check key attributes:
   - For ProductModule: same `code` (e.g., code="ITSC") supports deduplication
   - For CmdbModel: same `model_id` supports deduplication
   - For Component: same `component_name` supports deduplication
3. **Do NOT** deduplicate based solely on matching attributes if name and summary describe different concepts.
```

## 关键原则

| 优先级 | 证据 | 说明 |
|-------|-----|------|
| 1 | Name + Summary | 主要依据，如果 summary 明确描述同一概念则为重复 |
| 2 | Attributes | 辅助证据，支持去重判断但不作为唯一依据 |
| 3 | 禁止单独依赖 attributes | 如果 name/summary 描述不同概念，即使 code 相同也不应合并 |

## 修改文件清单

| 文件 | 修改内容 |
|-----|---------|
| `graphiti_core/prompts/dedupe_nodes.py` | `node()` 和 `nodes()` 明确 Deduplication Priority |

---
