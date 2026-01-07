# EntityEdge.get_by_uuid() 支持自定义边类型

## 问题背景

Graphiti 原生的 `EntityEdge.get_by_uuid()` 方法使用硬编码的 Cypher 查询只搜索 `RELATES_TO` 类型的边：

```cypher
MATCH (n:Entity)-[e:RELATES_TO {uuid: $uuid}]->(m:Entity)
```

当使用 EasyOps Schema 定义自定义边类型（如 `LIKES`, `WORKS_AT`, `MANAGES` 等）时，这些边会以自定义类型名称存储在 FalkorDB 中，而不是 `RELATES_TO`。这导致通过 UUID 查询边时返回 "Edge not found" 错误，即使边实际存在于图数据库中。

## 影响范围

- Edge History API (`GET /api/v1/edges/{uuid}/history`)
- 任何需要通过 UUID 获取 EntityEdge 的功能
- 时间语义测试中的边状态验证

## 修改内容

### 文件: `graphiti_core/edges.py`

**修改位置**: `EntityEdge.get_by_uuid()` 方法（约第 318-345 行）

**修改前**:
```python
@classmethod
async def get_by_uuid(cls, driver: GraphDriver, uuid: str):
    match_query = """
        MATCH (n:Entity)-[e:RELATES_TO {uuid: $uuid}]->(m:Entity)
    """
    # ...
```

**修改后**:
```python
@classmethod
async def get_by_uuid(cls, driver: GraphDriver, uuid: str):
    # EasyOps fix: Search for any edge type between Entity nodes, not just RELATES_TO
    # This supports custom edge types defined in Schema (e.g., LIKES, WORKS_AT)
    match_query = """
        MATCH (n:Entity)-[e]->(m:Entity)
        WHERE e.uuid = $uuid
    """
    if driver.provider == GraphProvider.KUZU:
        match_query = """
            MATCH (n:Entity)-[:RELATES_TO]->(e:RelatesToNode_ {uuid: $uuid})-[:RELATES_TO]->(m:Entity)
        """
    # ...
```

## 为什么不修改其他方法

其他方法（如 `get_by_uuids()`, `get_by_group_ids()`, `get_between_nodes()`, `get_by_node_uuid()`）仍然使用 `RELATES_TO`，因为：

1. 这些方法主要用于 Graphiti 内部的边处理逻辑
2. 在当前实现中，自定义边类型在写入时已被正确处理
3. `get_by_uuid()` 是唯一需要通过精确 UUID 查找单条边的方法，常用于外部 API（如 Edge History API）

如果后续发现其他方法也需要支持自定义边类型，可以按相同模式修改。

## 升级注意事项

升级 Graphiti 时需要手动合并此修改，检查点：

1. 查找 `EntityEdge.get_by_uuid()` 方法
2. 将 `MATCH (n:Entity)-[e:RELATES_TO {uuid: $uuid}]->(m:Entity)` 改为通用查询
3. 注意保持 KUZU provider 的特殊处理不变
