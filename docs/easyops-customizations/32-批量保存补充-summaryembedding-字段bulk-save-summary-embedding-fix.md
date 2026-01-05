# 32. 批量保存补充 summary_embedding 字段（Bulk Save Summary Embedding Fix）

## 问题背景

第 29 章节实现了"双向量 Max Score 策略"，EntityNode 新增了 `summary_embedding` 字段，用于分别存储 name 和 summary 的向量。

但在批量保存时（`add_nodes_and_edges_bulk_tx`），entity_data 字典只包含了 `name_embedding`，**遗漏了 `summary_embedding`**：

```python
# bulk_utils.py 第 1003-1015 行（修改前）
entity_data: dict[str, Any] = {
    'uuid': node.uuid,
    'name': name,
    'group_id': node.group_id,
    'summary': summary,
    'created_at': node.created_at,
    'name_embedding': node.name_embedding,
    # 缺失: 'summary_embedding': node.summary_embedding,
    'labels': list(set(node.labels + ['Entity'])),
    ...
}
```

**影响**：
- 所有通过 `add_episode_bulk` 批量导入的实体，`summary_embedding` 都是 NULL
- 双向量策略无法生效，搜索时只能使用 `name_embedding`
- 中文 summary 的语义无法被正确检索

**具体案例**：
- 查询 "What is the name of the father of Childericus?"
- 期望结果：Merovech（summary 包含 "father of Childeric I"）
- 实际结果：Merovech 排名第 40 位（只用 name_score=0.48，summary_score=0.0 因为向量为空）

## 解决方案

**文件**: `graphiti_core/utils/bulk_utils.py`

在 `add_nodes_and_edges_bulk_tx` 函数中的 entity_data 字典添加 `summary_embedding` 字段：

```python
entity_data: dict[str, Any] = {
    'uuid': node.uuid,
    'name': name,
    'group_id': node.group_id,
    'summary': summary,
    'created_at': node.created_at,
    'name_embedding': node.name_embedding,
    'summary_embedding': node.summary_embedding,  # EasyOps: 双向量策略
    'labels': list(set(node.labels + ['Entity'])),
    'reasoning': reasoning,
    # ...
}
```

## 对比 nodes.py 的 save() 方法

单个节点保存时（`nodes.py:536`）正确包含了 `summary_embedding`：

```python
entity_data: dict[str, Any] = {
    'uuid': self.uuid,
    'name': self.name,
    'name_embedding': self.name_embedding,
    'summary_embedding': self.summary_embedding,  # 正确包含
    # ...
}
```

批量保存应与单个保存保持一致。

## 效果

修复后，批量导入的实体会正确保存 `summary_embedding`，搜索时可以使用 max(name_score, summary_score) 策略。

| 指标 | 修复前 | 修复后 |
|-----|-------|-------|
| summary_embedding | NULL | 正确向量 |
| 中文 summary 语义检索 | ❌ 无效 | ✅ 正常 |
| Merovech 搜索排名 | 第 40 位 | 预期 Top 10 |

## 修改文件清单

| 文件 | 修改内容 |
|-----|---------|
| `graphiti_core/utils/bulk_utils.py` | `entity_data` 字典添加 `summary_embedding` 字段 |

## 升级注意事项

已导入的数据需要重新导入或运行批量更新脚本来生成 `summary_embedding`：

```bash
# 方法 1: 重新导入数据
# 清空 group_id 后重新运行导入

# 方法 2: 运行更新脚本
poetry run python scripts/update_all_embeddings.py <group_id>
```

---
