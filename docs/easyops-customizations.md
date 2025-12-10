# EasyOps Graphiti 自定义修改

本文档记录 EasyOps 对 Graphiti 的自定义修改，便于后续维护和升级时参考。

## 修改概览

| 文件 | 修改类型 | 说明 |
|------|---------|------|
| `graphiti_core/models/edges/edge_db_queries.py` | 新增函数 | 支持动态边类型的 Cypher 查询 |
| `graphiti_core/utils/bulk_utils.py` | 逻辑修改 | 按边类型分组保存 |
| `graphiti_core/utils/maintenance/edge_operations.py` | 逻辑修改 | 严格控制边类型，非 Schema 类型降级 |
| `graphiti_core/prompts/extract_nodes.py` | 新增 | Knowledge Graph Builder's Principles + filter_entities |
| `graphiti_core/utils/maintenance/node_operations.py` | 新增 | filter_extracted_nodes 函数 |
| `graphiti_core/llm_client/openai_generic_client.py` | 新增 | small_model 支持 |

---

## 1. Knowledge Graph Builder's Principles（实体抽取质量控制）

### 问题背景

原始 Graphiti 的实体抽取存在以下问题：
1. 抽取过多噪声实体（代码变量名、占位符、无意义的技术术语）
2. 实体分类准确率较低，大量实体被错误分类为 Feature 或 Entity
3. 原有的 reflexion 机制是"找遗漏"，而不是"过滤噪声"

### 解决方案

#### 1.1 添加 Knowledge Graph Builder's Principles

**文件**: `graphiti_core/prompts/extract_nodes.py`

在所有实体抽取提示词中添加四条核心原则：

```python
KNOWLEDGE_GRAPH_PRINCIPLES = """
## KNOWLEDGE GRAPH BUILDER'S PRINCIPLES (Must Follow)

You are building an enterprise knowledge graph. Every entity you extract will become a permanent node that other documents can reference and build relationships upon.

Before extracting any entity, verify it passes ALL four principles:

1. **Permanence Principle**: Only extract entities that have lasting value beyond this single document.
   Ask: "Will this entity still be meaningful and useful 6 months from now?"

2. **Connectivity Principle**: Only extract entities that can form meaningful relationships with other entities.
   Ask: "Can this entity connect to other concepts in a knowledge graph?"

3. **Independence Principle**: Only extract entities that are self-explanatory without the source document.
   Ask: "Would someone understand this entity name without reading the original text?"

4. **Domain Value Principle**: Only extract entities that represent real domain knowledge, not document artifacts.
   Ask: "Is this a concept a domain expert would recognize and care about?"

**EXTRACTION DECISION**: If uncertain about any principle, do NOT extract. It is better to miss an entity than to pollute the knowledge graph with noise.
"""
```

**应用于函数**:
- `extract_message()` - 会话消息抽取
- `extract_text()` - 文本抽取
- `extract_json()` - JSON 抽取

#### 1.2 添加 filter_entities 提示词

**文件**: `graphiti_core/prompts/extract_nodes.py`

新增 `filter_entities()` 函数，用于过滤不符合质量标准的实体：

```python
def filter_entities(context: dict[str, Any]) -> list[Message]:
    sys_prompt = """You are a knowledge graph quality reviewer. Your task is to identify entities that should NOT be in an enterprise knowledge graph.

Review each extracted entity against the Knowledge Graph Builder's Principles:

1. **Permanence Principle**: Does it have lasting value beyond this document?
2. **Connectivity Principle**: Can it meaningfully connect to other entities?
3. **Independence Principle**: Is the name self-explanatory without the source text?
4. **Domain Value Principle**: Does it represent real domain knowledge, not document artifacts?

An entity should be REMOVED if it fails ANY of these principles."""
    # ... user_prompt ...
```

**新增模型**:
```python
class EntitiesToFilter(BaseModel):
    entities_to_remove: list[str] = Field(
        ...,
        description='Names of entities that should be removed from the knowledge graph',
    )
    reasoning: str = Field(
        ...,
        description='Brief explanation of why these entities were flagged for removal',
    )
```

#### 1.3 添加 filter_extracted_nodes 函数

**文件**: `graphiti_core/utils/maintenance/node_operations.py`

新增 `filter_extracted_nodes()` 函数：

```python
async def filter_extracted_nodes(
    llm_client: LLMClient,
    episode: EpisodicNode,
    extracted_entities: list[ExtractedEntity],
    group_id: str | None = None,
) -> list[str]:
    """Filter out entities that don't meet knowledge graph quality standards.

    Uses the Knowledge Graph Builder's Principles to identify and remove:
    - Entities without lasting value (Permanence)
    - Entities that can't connect meaningfully (Connectivity)
    - Entities that aren't self-explanatory (Independence)
    - Document artifacts instead of domain knowledge (Domain Value)
    """
    # ... implementation ...
```

#### 1.4 修改 extract_nodes 流程

**文件**: `graphiti_core/utils/maintenance/node_operations.py`

在 `extract_nodes()` 函数中，反思循环结束后添加过滤步骤：

```python
# 原有的 reflexion 循环结束后...

# Filter entities using Knowledge Graph Builder's Principles
llm_start = time()
entities_to_remove = await filter_extracted_nodes(
    llm_client,
    episode,
    extracted_entities,
    episode.group_id,
)
llm_call_count += 1
perf_logger.info(f'[PERF]     └─ extract_nodes filter #{llm_call_count}: {(time() - llm_start)*1000:.0f}ms')

# Remove filtered entities
entities_to_remove_set = set(entities_to_remove)
extracted_entities = [e for e in extracted_entities if e.name not in entities_to_remove_set]
```

### 效果

- 减少噪声实体（代码变量、占位符等）的抽取
- 提高实体分类准确率
- 知识图谱质量更高，更有价值

---

## 2. 动态边类型支持

### 问题背景

原始 Graphiti 在批量保存边时，Cypher 查询硬编码了 `RELATES_TO` 作为关系类型：

```cypher
MERGE (source)-[r:RELATES_TO {uuid: edge.uuid}]->(target)
SET r = edge
```

这导致即使 LLM 正确识别了边类型（如 `DEPENDS_ON`），数据库中的关系类型仍然是 `RELATES_TO`，`edge.name` 只是一个属性而不是真正的关系类型。

### 具体表现

```
LLM 抽取结果:
  - source: agent (Component)
  - target: gateway (Component)
  - edge.name = "DEPENDS_ON"

数据库实际存储（修复前）:
  (agent)-[:RELATES_TO {name: "DEPENDS_ON"}]->(gateway)
           ↑ 关系类型错误

数据库实际存储（修复后）:
  (agent)-[:DEPENDS_ON {name: "DEPENDS_ON"}]->(gateway)
           ↑ 关系类型正确
```

### 修复方案

#### 2.1 新增函数 `get_entity_edge_save_bulk_query_by_type()`

**文件**: `graphiti_core/models/edges/edge_db_queries.py`

**位置**: 在 `get_entity_edge_save_bulk_query()` 函数之后新增

```python
def get_entity_edge_save_bulk_query_by_type(provider: GraphProvider, edge_type: str) -> str:
    """Generate edge save query for a specific edge type.

    This is necessary because Cypher doesn't support dynamic relationship types.
    We generate a query with the edge type hardcoded in the MERGE clause.
    """
    # Sanitize edge_type to prevent injection (only allow alphanumeric and underscore)
    safe_edge_type = ''.join(c for c in edge_type if c.isalnum() or c == '_')
    if not safe_edge_type:
        safe_edge_type = 'RELATES_TO'

    match provider:
        case GraphProvider.FALKORDB:
            return f"""
                UNWIND $entity_edges AS edge
                MATCH (source:Entity {{uuid: edge.source_node_uuid}})
                MATCH (target:Entity {{uuid: edge.target_node_uuid}})
                MERGE (source)-[r:{safe_edge_type} {{uuid: edge.uuid}}]->(target)
                SET r = edge
                SET r.fact_embedding = vecf32(edge.fact_embedding)
                WITH r, edge
                RETURN edge.uuid AS uuid
            """
        case GraphProvider.NEPTUNE:
            return f"""
                UNWIND $entity_edges AS edge
                MATCH (source:Entity {{uuid: edge.source_node_uuid}})
                MATCH (target:Entity {{uuid: edge.target_node_uuid}})
                MERGE (source)-[r:{safe_edge_type} {{uuid: edge.uuid}}]->(target)
                SET r = removeKeyFromMap(removeKeyFromMap(edge, "fact_embedding"), "episodes")
                SET r.fact_embedding = join([x IN coalesce(edge.fact_embedding, []) | toString(x) ], ",")
                SET r.episodes = join(edge.episodes, ",")
                RETURN edge.uuid AS uuid
            """
        case _:
            # For Neo4j and others
            return f"""
                UNWIND $entity_edges AS edge
                MATCH (source:Entity {{uuid: edge.source_node_uuid}})
                MATCH (target:Entity {{uuid: edge.target_node_uuid}})
                MERGE (source)-[e:{safe_edge_type} {{uuid: edge.uuid}}]->(target)
                SET e += edge
                WITH e, edge
                CALL db.create.setRelationshipVectorProperty(e, "fact_embedding", edge.fact_embedding)
                RETURN edge.uuid AS uuid
            """
```

**安全性**: 使用 `safe_edge_type` 过滤非法字符，防止 Cypher 注入。

#### 2.2 修改批量保存逻辑

**文件**: `graphiti_core/utils/bulk_utils.py`

**位置**: `add_nodes_and_edges_bulk_tx()` 函数末尾，替换原来的边保存逻辑

**修改前**:
```python
await tx.run(
    get_entity_edge_save_bulk_query(driver.provider),
    entity_edges=edges,
)
```

**修改后**:
```python
# Group edges by type and save each group with the correct relationship type
# This is necessary because Cypher doesn't support dynamic relationship types
edges_by_type: dict[str, list[dict[str, Any]]] = {}
for edge in edges:
    edge_type = edge.get('name', 'RELATES_TO')
    if edge_type not in edges_by_type:
        edges_by_type[edge_type] = []
    edges_by_type[edge_type].append(edge)

for edge_type, typed_edges in edges_by_type.items():
    query = get_entity_edge_save_bulk_query_by_type(driver.provider, edge_type)
    logger.info(f'[bulk_save] Saving {len(typed_edges)} edges of type {edge_type}')
    await tx.run(query, entity_edges=typed_edges)
```

**同时需要添加 import**:
```python
from graphiti_core.models.edges.edge_db_queries import (
    get_entity_edge_save_bulk_query,
    get_entity_edge_save_bulk_query_by_type,  # 新增
    get_episodic_edge_save_bulk_query,
)
```

### 验证方法

导入数据后，查询边类型分布：

```cypher
MATCH ()-[r]->()
RETURN type(r) as edge_type, count(*) as cnt
ORDER BY cnt DESC
```

修复前只会看到 `RELATES_TO` 和 `MENTIONS`，修复后会看到多种边类型。

---

## 3. 其他自定义修改

### 3.1 LLM Prompt 增加 reasoning 字段

**文件**:
- `graphiti_core/prompts/extract_nodes.py`
- `graphiti_core/prompts/extract_edges.py`
- `graphiti_core/prompts/dedupe_edges.py`

**目的**: 让 LLM 输出推理过程，提高抽取准确性（Chain of Thought）。

### 3.2 FalkorDB 字符串转义

**文件**: `graphiti_core/utils/bulk_utils.py`

**函数**: `_sanitize_string_for_falkordb()`

**目的**: 转义换行符等控制字符，防止 FalkorDB 查询解析错误。

### 3.3 非官方 OpenAI 兼容端点支持

**文件**: `graphiti_core/llm_client/openai_generic_client.py`

**目的**: 支持不支持 `response_format: json_schema` 的 OpenAI 兼容端点（如阿里 Qwen）。

### 3.4 严格控制边类型（非 Schema 边类型降级）

**文件**: `graphiti_core/utils/maintenance/edge_operations.py`

**位置**: `resolve_extracted_edge()` 函数中的边类型处理逻辑

**问题背景**:

原始 Graphiti 的边类型处理逻辑：
1. 如果是 Schema 允许的边类型 → 采用
2. 如果是 Schema 定义但不允许当前节点对 → 降级为 RELATES_TO
3. 如果是 LLM 自动生成的非 Schema 边类型 → **保留** ← 问题！

这导致 LLM 可能生成任意边类型名称（如 `DEVELOPED_IN`、`USES_PORT`），绕过 Schema 控制。

**修改**:

```python
# 修改前
elif not is_default_type:
    # Non-custom labels are allowed to pass through
    resolved_edge.name = fact_type  # 保留 LLM 生成的名称
    resolved_edge.attributes = {}

# 修改后
elif not is_default_type:
    # EasyOps customization: Non-schema edge types also degrade to DEFAULT
    resolved_edge.name = DEFAULT_EDGE_NAME  # 降级为 RELATES_TO
    resolved_edge.attributes = {}
```

**效果**: 只有 Schema 中明确定义的边类型才会被保留，其他一律降级为 `RELATES_TO`。

---

## 升级注意事项

升级 Graphiti 时，需要检查以下文件是否有冲突：

1. `graphiti_core/prompts/extract_nodes.py` - 确保 KNOWLEDGE_GRAPH_PRINCIPLES 和 filter_entities 函数存在
2. `graphiti_core/utils/maintenance/node_operations.py` - 确保 filter_extracted_nodes 函数和调用逻辑存在
3. `graphiti_core/models/edges/edge_db_queries.py` - 确保 `get_entity_edge_save_bulk_query_by_type()` 函数存在
4. `graphiti_core/utils/bulk_utils.py` - 确保边按类型分组保存的逻辑存在
5. `graphiti_core/utils/maintenance/edge_operations.py` - 确保非 Schema 边类型降级逻辑存在
6. 各 prompt 文件 - 确保 reasoning 字段的修改保留

建议使用 `git diff` 对比上游更新，手动合并自定义修改。

---

## 4. Small Model 支持

### 问题背景

Graphiti 的 `OpenAIClient` 支持 `small_model` 参数，可以为简单任务使用较小的模型以节省成本。但 Elevo Memory 使用的 `OpenAIGenericClient`（用于非官方 OpenAI 兼容端点）没有实现这个功能，导致 `model_size` 参数被忽略。

### 修改内容

#### 4.1 添加 _get_model_for_size 方法

**文件**: `graphiti_core/llm_client/openai_generic_client.py`

```python
def _get_model_for_size(self, model_size: ModelSize) -> str:
    """Get the appropriate model name based on the requested size."""
    if model_size == ModelSize.small:
        return self.small_model or self.model or DEFAULT_MODEL
    else:
        return self.model or DEFAULT_MODEL
```

#### 4.2 修改 _generate_response 使用该方法

```python
# 原来
response = await self.client.chat.completions.create(
    model=self.model or DEFAULT_MODEL,
    ...
)

# 修改后
model = self._get_model_for_size(model_size)
response = await self.client.chat.completions.create(
    model=model,
    ...
)
```

### 配置方法

在 `.env` 文件或环境变量中配置：

```bash
# 主模型（复杂任务）
OPENAI_MODEL=qwen3-235a22b-2507-local_2

# 小模型（简单任务，可选）
OPENAI_SMALL_MODEL=qwen3-235a22b-2507-local_2
```

如果不设置 `OPENAI_SMALL_MODEL`，系统会自动使用 `OPENAI_MODEL` 作为 fallback。
