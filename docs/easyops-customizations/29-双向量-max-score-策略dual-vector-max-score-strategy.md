# 29. 双向量 Max Score 策略（Dual-Vector Max Score Strategy）

## 问题背景

**语义稀释问题**：当英文实体名（如 `cmdb-history-jump`）与中文 summary 拼接生成单一 embedding 时，英文部分（语义密度低）会稀释中文部分（语义密度高）的表达能力。

**具体现象**：
- 实体 `cmdb-history-jump`（summary: "IT资源审计功能，需特性开关控制"）
- 用户查询："IT资源审计功能 特性开关"
- 预期排名：Top 10
- 实际排名：第 117 位

**根本原因**：
```
原始方案：embedding = embed(name + " " + summary)
         = embed("cmdb-history-jump IT资源审计功能...")

问题：英文 "cmdb-history-jump" 作为噪声，将向量推离纯中文查询向量的方向
```

## 解决方案

**双向量策略**：分别存储 `name_embedding` 和 `summary_embedding`，搜索时取最大相似度。

```
新方案：
  name_embedding = embed(name)           # 仅用英文名
  summary_embedding = embed(summary)     # 仅用中文描述

搜索时：
  name_score = cosine_sim(query_embedding, name_embedding)
  summary_score = cosine_sim(query_embedding, summary_embedding)
  final_score = max(name_score, summary_score)
```

## 修改详情

### 1. EntityNode 新增 summary_embedding 字段

**文件**: `graphiti_core/nodes.py`

```python
class EntityNode(Node):
    name_embedding: list[float] | None = Field(default=None, description='embedding of the name')
    summary_embedding: list[float] | None = Field(default=None, description='embedding of the summary for semantic search')
    # ...

    async def generate_name_embedding(self, embedder: EmbedderClient):
        """生成 name_embedding 和 summary_embedding（双向量策略）"""
        # name_embedding: 仅用 name 生成
        name_text = self.name.replace('\n', ' ')
        self.name_embedding = await embedder.create(input_data=[name_text])

        # summary_embedding: 用 summary 生成（如果有 summary）
        if self.summary:
            summary_text = self.summary.replace('\n', ' ')
            self.summary_embedding = await embedder.create(input_data=[summary_text])
        else:
            self.summary_embedding = None

        return self.name_embedding
```

同时修改 `create_entity_node_embeddings()` 批量生成函数：

```python
async def create_entity_node_embeddings(embedder: EmbedderClient, nodes: list[EntityNode]):
    """批量生成实体节点的 name_embedding 和 summary_embedding"""
    filtered_nodes = [node for node in nodes if node.name]
    if not filtered_nodes:
        return

    # 1. 生成 name_embedding（仅用 name）
    name_texts = [node.name for node in filtered_nodes]
    name_embeddings = await embedder.create_batch(name_texts)
    for node, name_embedding in zip(filtered_nodes, name_embeddings, strict=True):
        node.name_embedding = name_embedding

    # 2. 生成 summary_embedding（仅用 summary，如果有的话）
    nodes_with_summary = [node for node in filtered_nodes if node.summary]
    if nodes_with_summary:
        summary_texts = [node.summary for node in nodes_with_summary]
        summary_embeddings = await embedder.create_batch(summary_texts)
        for node, summary_embedding in zip(nodes_with_summary, summary_embeddings, strict=True):
            node.summary_embedding = summary_embedding
```

### 2. node_similarity_search 使用 Max Score

**文件**: `graphiti_core/search/search_utils.py`

```python
async def node_similarity_search(
    driver: GraphDriver,
    search_vector: list[float],
    search_filter: SearchFilters,
    group_ids: list[str] | None = None,
    limit=RELEVANT_SCHEMA_LIMIT,
    min_score: float = DEFAULT_MIN_SCORE,
) -> list[EntityNode]:
    # ...

    # 双向量 Max Score 策略：取 name_embedding 和 summary_embedding 的最大相似度
    name_score_expr = get_vector_cosine_func_query('n.name_embedding', search_vector_var, driver.provider)

    if driver.provider == GraphProvider.FALKORDB:
        # FalkorDB: 使用 CASE WHEN 处理 summary_embedding 可能为 NULL 的情况
        summary_score_expr = f"""
            CASE WHEN n.summary_embedding IS NOT NULL
                 THEN {get_vector_cosine_func_query('n.summary_embedding', search_vector_var, driver.provider)}
                 ELSE 0.0 END"""
        max_score_expr = f"""
            CASE WHEN ({name_score_expr}) > ({summary_score_expr})
                 THEN ({name_score_expr})
                 ELSE ({summary_score_expr}) END"""
    else:
        # Neo4j/Kuzu: 使用 coalesce 处理 NULL
        summary_score_expr = f"coalesce({get_vector_cosine_func_query('n.summary_embedding', search_vector_var, driver.provider)}, 0.0)"
        max_score_expr = f"CASE WHEN ({name_score_expr}) > ({summary_score_expr}) THEN ({name_score_expr}) ELSE ({summary_score_expr}) END"

    query = (
        """MATCH (n:Entity)"""
        + filter_query
        + """
        WITH n, """ + max_score_expr + """ AS score
        WHERE score > $min_score
        RETURN ..."""
    )
```

### 3. EntityNode.save() 保存 summary_embedding

**文件**: `graphiti_core/nodes.py`

```python
async def save(self, driver: GraphDriver):
    entity_data: dict[str, Any] = {
        'uuid': self.uuid,
        'name': self.name,
        'name_embedding': self.name_embedding,
        'summary_embedding': self.summary_embedding,  # 新增
        # ...
    }
```

## 批量更新现有数据

对于已有数据，需要运行批量更新脚本重新生成双向量：

**文件**: `scripts/update_all_embeddings.py`

```bash
# 更新所有实体的 embedding
poetry run python scripts/update_all_embeddings.py <group_id>

# 验证特定实体
poetry run python scripts/update_all_embeddings.py <group_id> --verify <entity_name> <query>
```

**注意事项**：
- FalkorDB 的 `vecf32` 属性更新必须先 `REMOVE` 再 `SET`
- 直接 `SET n.name_embedding = vecf32(...)` 不会生效
- 正确写法：`REMOVE n.name_embedding, n.summary_embedding SET n.name_embedding = vecf32(...)`

## 效果验证

| 指标 | 修复前 | 修复后 |
|-----|-------|-------|
| `cmdb-history-jump` 搜索排名 | 第 117 位 | 第 19 位 |
| name_score | - | 0.7071 |
| summary_score | - | 0.8022 |
| 最终得分 | 拼接向量相似度 | max(0.7071, 0.8022) = 0.8022 |

## 修改文件清单

| 文件 | 修改内容 |
|-----|---------|
| `graphiti_core/nodes.py` | 新增 `summary_embedding` 字段，修改 `generate_name_embedding()` 和 `create_entity_node_embeddings()` |
| `graphiti_core/search/search_utils.py` | `node_similarity_search()` 使用 max(name_score, summary_score) |
| `scripts/update_all_embeddings.py` | 批量更新脚本，支持 REMOVE + SET 模式更新 vecf32 |

## 升级注意事项

1. **数据迁移**：升级后需运行批量更新脚本为所有实体生成 `summary_embedding`
2. **存储增加**：每个有 summary 的实体会额外存储一个向量（约 3KB/实体，1536 维）
3. **查询兼容**：`summary_embedding` 为 NULL 时自动降级为仅使用 `name_embedding`

---
