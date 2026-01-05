# 33. FalkorDB 批量保存避免 embedding 存储为 List（Batch Save Vectorf32 Fix）

## 问题背景

FalkorDB 批量保存实体节点时，`summary_embedding` 被存储为 Python List 类型而非 FalkorDB 原生的 Vectorf32 类型，导致向量搜索时报错：

```
Type mismatch: expected Null or Vectorf32 but was List
```

**具体现象**：
- `name_embedding` 向量自相似度查询正常（返回 0.0）
- `summary_embedding` 向量自相似度查询失败（类型不匹配）

```cypher
-- 正常（name_embedding 是 Vectorf32）
MATCH (n:Entity) RETURN vec.cosineDistance(n.name_embedding, n.name_embedding) LIMIT 1
-- 返回: 0.0

-- 失败（summary_embedding 是 List）
MATCH (n:Entity) RETURN vec.cosineDistance(n.summary_embedding, n.summary_embedding) LIMIT 1
-- 错误: Type mismatch: expected Null or Vectorf32 but was List
```

## 根因分析

**文件**: `graphiti_core/models/nodes/node_db_queries.py`

原有的 FalkorDB 批量保存查询使用 `SET n = node` 一次性设置所有属性：

```cypher
UNWIND $nodes AS node
MERGE (n:Entity {uuid: node.uuid})
SET n:{label}
SET n = node                    -- 问题根源：这会把 embedding 设置为 List
WITH n, node
SET n.name_embedding = vecf32(node.name_embedding)
SET n.summary_embedding = CASE WHEN node.summary_embedding IS NOT NULL
                               THEN vecf32(node.summary_embedding) ELSE NULL END
RETURN n.uuid AS uuid
```

**问题**：
1. `SET n = node` 会设置 node 字典中的**所有**属性，包括 `name_embedding` 和 `summary_embedding`
2. 此时 embedding 被设置为 Python List 类型
3. 后续的 `SET n.name_embedding = vecf32(...)` 对于 `name_embedding` 能正常覆盖
4. 但 `summary_embedding` 虽然也尝试用 `vecf32()` 覆盖，FalkorDB 不允许将 List 类型属性覆盖为 Vectorf32 类型

**关键发现**：`name_embedding` 能正常工作是因为它在 `SET n = node` 之前就存在于 Cypher 查询参数中，而 FalkorDB 对已存在的 Vectorf32 属性可以重新赋值。但对于新设置的 List 属性，无法直接转换为 Vectorf32。

## 解决方案

**修改策略**：不使用 `SET n = node`，而是显式设置每个属性，确保 embedding 只通过 `vecf32()` 函数设置。

**修改前**：
```python
case GraphProvider.FALKORDB:
    queries = []
    for node in nodes:
        for label in node['labels']:
            queries.append(
                (
                    f"""
                    UNWIND $nodes AS node
                    MERGE (n:Entity {{uuid: node.uuid}})
                    SET n:{label}
                    SET n = node
                    WITH n, node
                    SET n.name_embedding = vecf32(node.name_embedding)
                    SET n.summary_embedding = CASE WHEN node.summary_embedding IS NOT NULL THEN vecf32(node.summary_embedding) ELSE NULL END
                    RETURN n.uuid AS uuid
                    """,
                    {'nodes': [node]},
                )
            )
    return queries
```

**修改后**：
```python
case GraphProvider.FALKORDB:
    queries = []
    for node in nodes:
        for label in node['labels']:
            # EasyOps fix: Set properties individually to avoid List type for embeddings
            # Using SET n = node would set embeddings as List, then vecf32() would fail
            # because FalkorDB doesn't allow overwriting List with Vectorf32
            queries.append(
                (
                    f"""
                    UNWIND $nodes AS node
                    MERGE (n:Entity {{uuid: node.uuid}})
                    SET n:{label}
                    SET n.uuid = node.uuid,
                        n.name = node.name,
                        n.group_id = node.group_id,
                        n.summary = node.summary,
                        n.created_at = node.created_at,
                        n.labels = node.labels,
                        n.reasoning = node.reasoning,
                        n.type_scores = node.type_scores,
                        n.type_confidence = node.type_confidence
                    WITH n, node
                    SET n.name_embedding = vecf32(node.name_embedding)
                    SET n.summary_embedding = CASE WHEN node.summary_embedding IS NOT NULL THEN vecf32(node.summary_embedding) ELSE NULL END
                    RETURN n.uuid AS uuid
                    """,
                    {'nodes': [node]},
                )
            )
    return queries
```

## 效果

修复后，`summary_embedding` 正确存储为 Vectorf32 类型，双向量搜索策略正常工作：

| 指标 | 修复前 | 修复后 |
|-----|-------|-------|
| `summary_embedding` 类型 | List（错误） | Vectorf32（正确） |
| 向量自相似度查询 | 类型不匹配错误 | 正常返回 0.0 |
| 双向量 Max Score 搜索 | 失败 | 正常工作 |
| Merovech 搜索排名（父亲问题） | N/A（搜索失败） | Top 10 |

## 验证方法

```cypher
-- 验证 summary_embedding 类型正确
MATCH (n:Entity)
WHERE n.summary_embedding IS NOT NULL
RETURN n.name, vec.cosineDistance(n.summary_embedding, n.summary_embedding) AS self_sim
LIMIT 5

-- 预期结果：self_sim 全部为 0.0（完全相似）
```

## 修改文件清单

| 文件 | 修改内容 |
|-----|---------|
| `graphiti_core/models/nodes/node_db_queries.py` | FalkorDB 分支改为显式设置属性，避免 `SET n = node` |

## 升级注意事项

已导入的数据如果存在 `summary_embedding` 为 List 类型的问题，需要：

1. **删除旧图并重新导入**（推荐）：
   ```bash
   # 删除图
   curl -X DELETE http://localhost:8000/api/v1/graphs/{group_id}

   # 重新导入
   curl -X POST http://localhost:8000/api/v1/episodes/bulk -d @data.json
   ```

2. **或运行批量更新脚本**：
   ```bash
   poetry run python scripts/update_all_embeddings.py <group_id>
   ```

---
