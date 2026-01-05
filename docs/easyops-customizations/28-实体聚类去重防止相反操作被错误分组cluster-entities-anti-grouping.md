# 28. 实体聚类去重：防止相反操作被错误分组（Cluster Entities Anti-Grouping）

## 问题背景

批量去重使用 `cluster_entities` 提示词时，LLM 输出自相矛盾的结果：

**具体案例**：

```json
{
  "entity_ids": [14, 33],
  "canonical_id": 14,
  "reasoning": "data_backup与data_restore为互为反向操作，分别用于备份与恢复easy_core数据，功能相反，不视为同一实体。"
}
```

**问题**：reasoning 明确说"功能相反，不视为同一实体"，但却将两个实体分到同一组，导致 `data_backup` 和 `data_restore` 被错误合并。

## 解决方案

**文件**: `graphiti_core/prompts/dedupe_nodes.py`

在 `cluster_entities()` 函数中增强提示词：

1. **添加 CRITICAL 规则**：明确列出不应分组的情况
2. **增加一致性检查要求**：reasoning 必须与分组决策一致

```python
**CRITICAL - DO NOT GROUP THESE**:
- Inverse/opposite operations: backup vs restore, create vs delete, import vs export
- Related but distinct concepts: request vs response, start vs stop, enable vs disable
- Different instances: server1 vs server2, config_a vs config_b
- Parent-child relationships: system vs subsystem, module vs submodule

If entities have OPPOSITE or COMPLEMENTARY functions, they are DIFFERENT entities - put each in its OWN group.

**RULES**:
...
5. **Your reasoning MUST be consistent with the grouping** - if reasoning says entities are "not the same" or "different", they must be in SEPARATE groups
```

## 效果

| 实体对 | 修复前 | 修复后 |
|-------|-------|-------|
| `data_backup` vs `data_restore` | 错误分到同一组 | 各自独立分组 |
| `create_xxx` vs `delete_xxx` | 可能被错误合并 | 各自独立分组 |
| `import_data` vs `export_data` | 可能被错误合并 | 各自独立分组 |

## 修改文件清单

| 文件 | 修改内容 |
|-----|---------|
| `graphiti_core/prompts/dedupe_nodes.py` | `cluster_entities()` 添加 CRITICAL 规则和 reasoning 一致性要求 |

---
