# 22. 批量保存补充 type_scores 和 type_confidence 字段

## 问题背景

LLM 抽取实体时使用 `extract_text_with_scores` 或 `extract_message_with_scores` 提示词，会返回每个实体类型的置信度分数。这些分数被正确存储到 `EntityNode` 对象中，但在批量保存时丢失。

问题链路：
1. `node_operations.py:459-460` - 正确设置 `type_scores` 和 `type_confidence` 到 EntityNode
2. `bulk_utils.py:463-472` - 构建 `entity_data` 时**遗漏**了这两个字段
3. 结果：数据库中 `type_scores` 和 `type_confidence` 始终为 null

## 解决方案

**文件**: `graphiti_core/utils/bulk_utils.py`

在 `_prepare_bulk_data()` 函数中的 `entity_data` 字典补充这两个字段：

```python
entity_data: dict[str, Any] = {
    'uuid': node.uuid,
    'name': name,
    'group_id': node.group_id,
    'summary': summary,
    'created_at': node.created_at,
    'name_embedding': node.name_embedding,
    'labels': list(set(node.labels + ['Entity'])),
    'reasoning': reasoning,
    # EasyOps: Save type classification scores (same as nodes.py save())
    'type_scores': json.dumps(node.type_scores) if node.type_scores else None,
    'type_confidence': node.type_confidence,
}
```

## 对比 nodes.py 的 save() 方法

`nodes.py:502-503` 中的实现作为参考：

```python
'type_scores': json.dumps(self.type_scores) if self.type_scores else None,
'type_confidence': self.type_confidence,
```

## 修改文件清单

| 文件 | 修改内容 |
|-----|---------|
| `graphiti_core/utils/bulk_utils.py` | `entity_data` 补充 `type_scores` 和 `type_confidence` 字段 |

---
