# 30. BFS 遍历所有边类型（BFS Traverse All Edge Types）

## 问题背景

EasyOps 在第 2 章节"动态边类型支持"中修改了边的保存逻辑，使 LLM 抽取的边类型（如 `CREATED`、`DIRECTED`、`MANAGES`）能正确保存到数据库，而非统一使用 `RELATES_TO`。

但 BFS 搜索代码没有相应更新，仍然硬编码遍历 `[:RELATES_TO|MENTIONS]`：

```cypher
-- 修改前（只遍历这两种边类型）
MATCH path = (origin {uuid: origin_uuid})-[:RELATES_TO|MENTIONS*1..{depth}]->(:Entity)
MATCH (n:Entity)-[e:RELATES_TO {uuid: rel.uuid}]-(m:Entity)
```

这导致通过其他边类型连接的实体无法被 BFS 发现。

**具体案例**：

问题："Who wrote Tom Vaughan's 2008 film?"
期望答案："Dana Fox"

图中存在的边：
```
Tom Vaughan --[CREATED]--> What Happens in Vegas
Dana Fox --[CREATED]--> What Happens in Vegas
```

但因为 BFS 只遍历 `RELATES_TO|MENTIONS`，从 `Tom Vaughan` 出发无法发现 `What Happens in Vegas` 节点，进而无法通过关系推理找到 `Dana Fox`。

## 解决方案

修改 BFS 查询，遍历所有边类型：

```cypher
-- 修改后（遍历任意边类型）
MATCH path = (origin {uuid: origin_uuid})-[*1..{depth}]->(:Entity)
MATCH (n:Entity)-[e {uuid: rel.uuid}]-(m:Entity)
```

**文件**: `graphiti_core/search/search_utils.py`

**Neptune 分支**（第 501、504 行）：

```python
# 修改前
MATCH path = (origin {uuid: origin_uuid})-[:RELATES_TO|MENTIONS *1..{bfs_max_depth}]->(n:Entity)
MATCH (n:Entity)-[e:RELATES_TO {uuid: rel.uuid}]-(m:Entity)

# 修改后
MATCH path = (origin {uuid: origin_uuid})-[*1..{bfs_max_depth}]->(n:Entity)
MATCH (n:Entity)-[e {uuid: rel.uuid}]-(m:Entity)
```

**FalkorDB/Neo4j 分支**（第 528、530 行）：

```python
# 修改前
MATCH path = (origin {uuid: origin_uuid})-[:RELATES_TO|MENTIONS*1..{bfs_max_depth}]->(:Entity)
MATCH (n:Entity)-[e:RELATES_TO {uuid: rel.uuid}]-(m:Entity)

# 修改后
MATCH path = (origin {uuid: origin_uuid})-[*1..{bfs_max_depth}]->(:Entity)
MATCH (n:Entity)-[e {uuid: rel.uuid}]-(m:Entity)
```

## 效果

| 场景 | 修复前 | 修复后 |
|-----|-------|-------|
| 通过 CREATED 边连接的实体 | ❌ 无法发现 | ✅ 正常遍历 |
| 通过 DIRECTED 边连接的实体 | ❌ 无法发现 | ✅ 正常遍历 |
| 通过 MANAGES 边连接的实体 | ❌ 无法发现 | ✅ 正常遍历 |
| 多跳推理（如 Person → Movie → Writer） | ❌ 中断 | ✅ 完整路径 |

## 注意事项

1. **Kuzu 分支未修改**：Kuzu 使用特殊的中间节点存储边（`RelatesToNode_`），其 `RELATES_TO` 是遍历边而非业务边类型，保持原有逻辑
2. **性能影响**：遍历所有边类型可能略微增加查询范围，但由于有 `depth` 限制和结果去重，影响可控

## 修改文件清单

| 文件 | 修改内容 |
|-----|---------|
| `graphiti_core/search/search_utils.py` | Neptune 分支 BFS 查询改为遍历所有边类型 |
| `graphiti_core/search/search_utils.py` | FalkorDB/Neo4j 分支 BFS 查询改为遍历所有边类型 |

---
