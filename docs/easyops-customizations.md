# EasyOps Graphiti 自定义修改

本文档记录 EasyOps 对 Graphiti 的自定义修改，便于后续维护和升级时参考。

## 修改概览

| 文件 | 修改类型 | 说明 |
|------|---------|------|
| `graphiti_core/models/edges/edge_db_queries.py` | 新增函数 | 支持动态边类型的 Cypher 查询 |
| `graphiti_core/models/edges/edge_db_queries.py` | 逻辑修改 | MERGE 按节点对合并，避免重复边 |
| `graphiti_core/utils/bulk_utils.py` | 逻辑修改 | 按边类型分组保存 |
| `graphiti_core/utils/bulk_utils.py` | 新增函数 | 批量去重第三轮 LLM 语义去重 |
| `graphiti_core/utils/bulk_utils.py` | **架构重构** | **聚类去重：避免自匹配问题 + 按类型分组 + Map-Reduce** |
| `graphiti_core/prompts/dedupe_nodes.py` | **新增** | **EntityClustering 模型 + cluster_entities 提示词** |
| `graphiti_core/graphiti.py` | **性能优化** | **移除 resolve_extracted_nodes 调用，改用 semantic_dedupe_nodes_bulk** |
| `graphiti_core/graphiti.py` | **性能优化** | **移除全局 DB 候选收集，由 Map 阶段按批处理** |
| `graphiti_core/utils/maintenance/edge_operations.py` | 逻辑修改 | 严格控制边类型，非 Schema 类型降级 |
| `graphiti_core/prompts/extract_nodes.py` | 新增 | Knowledge Graph Builder's Principles + filter_entities |
| `graphiti_core/utils/maintenance/node_operations.py` | 新增 | filter_extracted_nodes 函数（接收 EntityNode 含 summary） |
| `graphiti_core/graphiti.py` | 逻辑修改 | **Filter 步骤移到 extract_attributes 之后（获取 summary 后再过滤）** |
| `graphiti_core/llm_client/openai_generic_client.py` | 新增 | small_model 支持 |
| `graphiti_core/utils/maintenance/node_operations.py` | 逻辑修改 | 候选搜索按同实体类型过滤 |
| `graphiti_core/utils/maintenance/edge_operations.py` | 逻辑修改 | 候选搜索按同边类型过滤 |
| `graphiti_core/search/search_config.py` | 参数修改 | DEFAULT_SEARCH_LIMIT 从 10 改为 20 |
| `graphiti_core/prompts/dedupe_edges.py` | 新增字段 | EdgeDuplicate 添加 merged_fact 字段 |
| `graphiti_core/utils/maintenance/edge_operations.py` | 逻辑修改 | 边去重时合并 fact 和属性 |
| `graphiti_core/utils/maintenance/edge_operations.py` | 逻辑修改 | **边去重基于 (source, target, edge_type) 而非 fact** |
| `graphiti_core/prompts/dedupe_nodes.py` | 新增字段 | NodeDuplicate 添加 reasoning 字段 |
| `graphiti_core/prompts/dedupe_nodes.py` | Prompt优化 | 实体去重要求输出 reasoning，entity_type_definitions 独立 |
| `graphiti_core/utils/maintenance/node_operations.py` | 逻辑修改 | `_resolve_with_llm()` 独立提取类型定义，添加 reasoning 日志 |
| `graphiti_core/utils/maintenance/node_operations.py` | 容错处理 | **属性抽取枚举验证容错，移除无效字段而非崩溃** |
| `graphiti_core/prompts/extract_nodes.py` | Prompt优化 | **filter_entities 增加 Schema 实体类型上下文 + summary** |
| `graphiti_core/utils/maintenance/node_operations.py` | 参数传递 | filter_extracted_nodes 传入 entity_types_context |
| `graphiti_core/prompts/dedupe_nodes.py` | Prompt优化 | **基于属性的去重增强（code/model_id 相同即重复）** |
| `graphiti_core/prompts/extract_nodes.py` | 新增模型 | EntityReclassification 模型支持类型重新分类 |
| `graphiti_core/prompts/extract_nodes.py` | Prompt优化 | **filter_entities 支持类型重新验证和重新分类** |
| `graphiti_core/utils/maintenance/node_operations.py` | 返回类型 | filter_extracted_nodes 返回 (to_remove, to_reclassify) 元组 |
| `graphiti_core/graphiti.py` | 逻辑修改 | **处理误分类实体的类型更正** |
| `graphiti_core/prompts/extract_nodes.py` | 新增模型 | **两步验证: EntityValidationItem/Result** |
| `graphiti_core/prompts/extract_nodes.py` | 新增函数 | **validate_entity_types() 提示词（Step 1）** |
| `graphiti_core/utils/maintenance/node_operations.py` | 重构 | **filter_extracted_nodes 两步验证 + 批量并行（5实体/批）** |
| `graphiti_core/utils/maintenance/node_operations.py` | Bug修复 | **Step 2 不传 validation_reason 避免 LLM 偏向** |
| `graphiti_core/graphiti.py` | 逻辑修改 | **重新分类时同时更新 node.reasoning** |
| `graphiti_core/graphiti.py` | Bug修复 | **重新分类时清理旧类型属性，防止属性污染** |
| `graphiti_core/utils/bulk_utils.py` | Bug修复 | **批量保存补充 type_scores 和 type_confidence 字段** |
| `graphiti_core/prompts/extract_nodes.py` | 性能优化 | **实体类型打分从全量改为 Top 3 候选，减少 token 消耗** |
| `graphiti_core/prompts/extract_nodes.py` | 新增函数 | **resolve_ambiguous_types() 批量二次推理** |
| `graphiti_core/utils/maintenance/node_operations.py` | 逻辑修改 | **两阶段类型分类：Top 3 打分 + 歧义消解** |
| `graphiti_core/utils/maintenance/node_operations.py` | 性能优化 | **实体去重分批并行处理，避免 LLM 输出截断** |
| `graphiti_core/helpers.py` | Bug修复 | **lucene_sanitize 添加反引号转义，修复 RediSearch 语法错误** |
| `graphiti_core/utils/bulk_utils.py` | **Bug修复** | **批量去重传递 entity_type_definitions，启用别名匹配** |
| `graphiti_core/prompts/dedupe_nodes.py` | **Prompt增强** | **nodes() 添加别名匹配指导，识别 "或" 分隔的别名** |
| `graphiti_core/prompts/extract_nodes.py` | **功能增强** | **反思环节增加负向反思：识别不该抽取的实体、类型错误的实体** |
| `graphiti_core/prompts/extract_edges.py` | **功能增强** | **反思环节增加负向反思：识别不正确的关系、类型错误的关系** |
| `graphiti_core/utils/maintenance/node_operations.py` | 逻辑修改 | **extract_nodes_reflexion 处理负向反思结果** |
| `graphiti_core/utils/maintenance/edge_operations.py` | 逻辑修改 | **extract_edges 反思处理负向反思结果** |
| `graphiti_core/prompts/extract_nodes.py` | **性能优化** | **批量 Summary 提取：EntitySummaries 模型 + extract_summaries_bulk 提示词** |
| `graphiti_core/utils/maintenance/node_operations.py` | **性能优化** | **_extract_entity_summaries_bulk 批量提取 + extract_attributes_from_nodes 改用批量** |
| `scripts/cleanup_duplicate_nodes.py` | **新增脚本** | **离线清理重复节点：用 LLM 识别并合并数据库中的历史重复数据** |

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
    nodes: list[EntityNode],  # 接收 EntityNode（已有 summary）
    group_id: str | None = None,
    entity_types_context: list[dict] | None = None,
) -> list[str]:
    """Filter out entities that don't meet knowledge graph quality standards.

    Uses the Knowledge Graph Builder's Principles to identify and remove:
    - Entities without lasting value (Permanence)
    - Entities that can't connect meaningfully (Connectivity)
    - Entities that aren't self-explanatory (Independence)
    - Document artifacts instead of domain knowledge (Domain Value)

    If entity_types_context is provided, entities matching valid types will be preserved.
    """
    # Build entity info with summary for better filtering decisions
    entities_info = []
    for node in nodes:
        info = {'name': node.name}
        if node.summary:
            info['summary'] = node.summary
        # Include entity type for context
        specific_type = next((l for l in node.labels if l != 'Entity'), None)
        if specific_type:
            info['type'] = specific_type
        entities_info.append(info)
    # ... rest of implementation ...
```

#### 1.4 Filter 步骤位置：extract_attributes 之后

**重要变更**：Filter 步骤从 `extract_nodes()` 移到了 `graphiti.py` 的 `extract_attributes_from_nodes()` 之后。

**原因**：
- `extract_attributes_from_nodes` 会填充实体的 `summary` 属性
- Filter 需要 summary 来更准确地判断实体是否应该保留
- 在 summary 填充之前 filter 会导致误删合法实体

**文件**: `graphiti_core/graphiti.py`

**单 Episode 流程**（`add_episode` 方法）：

```python
# 并行执行 resolve_edges 和 extract_attributes
(resolved_edges, invalidated_edges), hydrated_nodes = await semaphore_gather(
    resolve_edges_task, extract_attrs_task
)

# Filter entities using Knowledge Graph Builder's Principles (after attributes/summary extracted)
entity_types_context = [...]  # 构建实体类型上下文
entities_to_remove = await filter_extracted_nodes(
    self.llm_client,
    episode,
    hydrated_nodes,  # 已经有 summary
    group_id,
    entity_types_context,
)
if entities_to_remove:
    # Filter nodes and their related edges
    hydrated_nodes = [n for n in hydrated_nodes if n.name not in entities_to_remove_set]
    # Also filter edges that reference removed nodes
    resolved_edges = [e for e in resolved_edges if ...]
```

**批量 Episode 流程**（`_resolve_nodes_and_edges_bulk` 方法）：

```python
# Extract attributes for resolved nodes
hydrated_nodes_results = await semaphore_gather(
    *[extract_attributes_from_nodes(...) for episode, previous_episodes in episode_context]
)

# Filter each episode's nodes in parallel
filter_results = await semaphore_gather(
    *[filter_extracted_nodes(self.llm_client, episode, hydrated_nodes_results[i], ...) for i, ...]
)

# Collect all entities to remove and filter
all_entities_to_remove = set()
for entities_to_remove in filter_results:
    all_entities_to_remove.update(entities_to_remove)

if all_entities_to_remove:
    # Filter nodes and edges
    ...
```

### 效果

- 减少噪声实体（代码变量、占位符等）的抽取
- **利用 summary 信息做更准确的过滤决策**
- 匹配 Schema 实体类型的实体不会被误删
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

### 3.1 LLM Prompt 增加 reasoning 字段（实体抽取入库）

**文件**:
- `graphiti_core/prompts/extract_nodes.py`
- `graphiti_core/prompts/extract_edges.py`
- `graphiti_core/prompts/dedupe_edges.py`
- `graphiti_core/nodes.py` - EntityNode 模型增加 reasoning 字段
- `graphiti_core/utils/maintenance/node_operations.py` - 提取 reasoning 并保存
- `graphiti_core/utils/bulk_utils.py` - 批量保存包含 reasoning

**目的**: 让 LLM 输出推理过程，提高抽取准确性（Chain of Thought），并将实体的 reasoning 入库供后续分析和调试。

**修改的模型**:

| 文件 | 模型 | 新增字段 |
|-----|------|---------|
| `extract_nodes.py` | `ExtractedEntity` | `reasoning: str` - 解释为什么抽取该实体及类型选择原因 |
| `extract_nodes.py` | `EntitiesToFilter` | `reasoning: str` - 解释为什么过滤这些实体 |
| `extract_edges.py` | `Edge` | `reasoning: str` - 解释为什么抽取该关系及类型选择原因 |
| `dedupe_edges.py` | `EdgeDuplicate` | `fact_type_reasoning: str` - 解释边类型分类的原因 |
| `nodes.py` | `EntityNode` | `reasoning: str \| None` - LLM 的分类推理过程 |

**实体 reasoning 入库说明**:
- 实体的 reasoning 字段会持久化到图数据库的节点属性中
- 查询实体时从 `properties(n)` 中提取 reasoning 并返回给前端
- 边的 reasoning 不入库，仅用于提高 LLM 输出质量

**关键修改点**:
1. `EntityNode` 模型新增 `reasoning: str | None` 字段
2. `extract_nodes()` 函数在创建新节点时传递 reasoning
3. `add_nodes_and_edges_bulk_tx()` 在保存时包含 reasoning
4. `get_entity_node_from_record()` 从 attributes 中提取 reasoning（因为使用 `properties(n) AS attributes` 返回所有节点属性）

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

---

## 5. 批量去重 LLM 语义去重（Semantic Dedup）

### 问题背景

批量导入时（`add_episode_bulk`），同一批次内的实体去重存在问题：

1. **确定性匹配局限**：精确字符串匹配和 MinHash Jaccard ≥ 0.9 无法处理语义同义词
2. **中英文同义词**：如 `EasyITSM` vs `IT服务中心` vs `EasyITSC`（同一产品的不同名称）
3. **Summary 时机问题**：原有的批量去重在 `extract_attributes_from_nodes` 之前执行，此时 summary 为空

导致同一实体被创建为多个节点。

### 解决方案

**文件**:
- `graphiti_core/utils/bulk_utils.py` - 新增函数
- `graphiti_core/graphiti.py` - 调用入口

在 `extract_attributes_from_nodes` 之后执行 LLM 语义去重，此时 summary 和 attributes 已填充。

#### 5.1 新增辅助函数

```python
def _get_entity_type_label(node: EntityNode) -> str:
    """获取实体的具体类型标签（非 'Entity'）"""

async def _resolve_batch_with_llm(...) -> list[tuple[EntityNode, EntityNode]]:
    """调用 LLM 识别语义重复，返回 (source, canonical) 节��对"""

def _merge_node_into_canonical(source: EntityNode, canonical: EntityNode) -> None:
    """合并 source 的 summary 和 attributes 到 canonical"""
```

#### 5.2 新增主函数

```python
async def semantic_dedupe_nodes_bulk(
    clients: GraphitiClients,
    nodes: list[EntityNode],
    entity_types: dict[str, type[BaseModel]] | None = None,
) -> list[EntityNode]:
    """
    在 extract_attributes_from_nodes 之后执行 LLM 语义去重。
    按实体类型分组，对同类型实体调用 LLM 检查是否为语义重复。
    发现重复时合并 summary 和 attributes。
    """
```

#### 5.3 调用位置

```python
# graphiti.py add_episode_bulk 中
(final_hydrated_nodes, ...) = await self._resolve_nodes_and_edges_bulk(...)

# EasyOps: 在 attributes 提取后执行语义去重
final_hydrated_nodes = await semantic_dedupe_nodes_bulk(
    self.clients, final_hydrated_nodes, entity_types
)

# 保存到图数据库
await add_nodes_and_edges_bulk(...)
```

### 合并逻辑

发现重复时，将 source 节点的信息合并到 canonical 节点：

```python
def _merge_node_into_canonical(source, canonical):
    # Summary: 拼接（如果都有且不重复）
    if source.summary and canonical.summary:
        if source.summary not in canonical.summary:
            canonical.summary = f"{canonical.summary} {source.summary}"
    elif source.summary:
        canonical.summary = source.summary

    # Attributes: source 填充 canonical 缺失的字段
    for key, value in source.attributes.items():
        if key not in canonical.attributes or not canonical.attributes[key]:
            canonical.attributes[key] = value
```

### 日志输出

- `[semantic_dedup] Checking X ProductModule entities for semantic duplicates`
- `[batch_dedup_llm] Sending 1 nodes to LLM against Y candidates`
- `[batch_dedup_llm] Duplicate found: "EasyITSM" -> "IT服务中心"`
- `[semantic_dedup] Merged "EasyITSM" into "IT服务中心"`

### 效果

- 批量导入时能正确识别语义同义词
- 有 summary 信息帮助 LLM 判断
- 合并重复实体的知识，不丢失信息

### 合并逻辑（_merge_node_into_canonical）

合并时会合并以下字段：
- **summary**: 拼接（如果都有且不重复）
- **attributes**: source 填充 canonical 缺失的字段
- **reasoning**: 拼接（用 `\n---\n` 分隔，保留所有推理过程）

---

## 6. 边去重时合并 Fact（Edge Fact Merging）

### 问题背景

原始 Graphiti 的边去重逻辑存在问题：
1. 当检测到重复边时，直接丢弃新边的 fact，只保留旧边的 fact
2. 不同语言或不同表述的相同关系没有被识别为重复
3. 重复边的属性（valid_at, invalid_at, attributes）没有合并
4. **关键问题**：同一对节点之间、同一边类型的边，即使 fact 描述不同，也会创建多条重复边

### 解决方案

#### 7.1 添加 merged_fact 字段

**文件**: `graphiti_core/prompts/dedupe_edges.py`

在 `EdgeDuplicate` 模型中添加 `merged_fact` 字段：

```python
class EdgeDuplicate(BaseModel):
    duplicate_facts: list[int] = Field(...)
    merged_fact: str | None = Field(
        default=None,
        description='When duplicate_facts is not empty, provide a merged fact that combines '
        'the semantic meaning of both the NEW FACT and the EXISTING FACT(s)...',
    )
    contradicted_facts: list[int] = Field(...)
    fact_type: str = Field(...)
    fact_type_reasoning: str = Field(...)
```

#### 7.2 更新提示词

**文件**: `graphiti_core/prompts/dedupe_edges.py`

在 `resolve_edge` 提示词中增加：

1. **重复检测增强**：明确"不同语言、不同措辞表达相同关系"也是重复
2. **Fact 合并任务**：当检测到重复时，LLM 必须返回合并后的 fact
3. **合并指南**：
   - 合并两个 fact 的语义信息
   - 去除冗余
   - 如果是不同语言，选择更详细或更一致的语言

#### 7.3 确定性边去重（2025-12-12 优化）

**核心改进**：边去重从"LLM 判断是否重复"改为"程序确定性判断 + LLM 合并 fact"。

**原理**：`(source_uuid, target_uuid, edge_type)` 相同的边就是同一条边，这是确定性逻辑，不需要 LLM 判断。LLM 只负责合并不同表述的 fact。

**文件**: `graphiti_core/utils/maintenance/edge_operations.py`

**批内去重** - `resolve_extracted_edges()` 函数：

```python
# 原来：基于 (source, target, fact) 去重
key = (
    edge.source_node_uuid,
    edge.target_node_uuid,
    _normalize_string_exact(edge.fact),  # fact 不同就认为是不同的边
)

# 修改后：基于 (source, target, edge_type) 去重
key = (
    edge.source_node_uuid,
    edge.target_node_uuid,
    edge.name,  # edge_type 相同就是同一条边
)
if key not in seen:
    seen[key] = edge
    facts_to_merge[edge.uuid] = [edge.fact]
else:
    # 同一对节点的同类型边，收集 fact 待合并
    canonical_edge = seen[key]
    facts_to_merge[canonical_edge.uuid].append(edge.fact)
    # 合并 episode 引用
    for ep_uuid in edge.episodes:
        if ep_uuid not in canonical_edge.episodes:
            canonical_edge.episodes.append(ep_uuid)
```

**与数据库已有边去重** - `resolve_extracted_edge()` 函数的快速路径：

```python
# 原来：基于 (source, target, fact) 精确匹配
if _normalize_string_exact(edge.fact) == normalized_fact:
    return resolved, [], []

# 修改后：基于 (source, target, edge_type) 匹配
if edge.name == extracted_edge.name:  # 同边类型 = 重复
    resolved = edge
    resolved.episodes.append(episode.uuid)
    return resolved, [extracted_edge], []  # 标记为重复，调用方处理 fact 合并
```

**效果**：
- 同一对节点间、同一边类型的边，无论 fact 如何不同，都会被识别为同一条边
- 多条 fact 描述会被合并（LLM 合并或拼接）
- 避免创建语义重复的边

#### 7.4 修改合并逻辑

**文件**: `graphiti_core/utils/maintenance/edge_operations.py`

修改 `resolve_extracted_edge` 函数中的合并逻辑，支持两种情况：

```python
for duplicate_fact_id in duplicate_fact_ids:
    existing_edge = related_edges[duplicate_fact_id]
    merged_fact = response_object.merged_fact
    if merged_fact and merged_fact.strip():
        # 使用 LLM 返回的合并 fact
        existing_edge.fact = merged_fact.strip()
        existing_edge.fact_embedding = None
        logger.info(f'[edge_dedup] Updated fact for edge {existing_edge.uuid} with LLM-merged content')
    else:
        # 强制合并场景：LLM 没有返回 merged_fact，手动拼接
        new_fact = extracted_edge.fact.strip()
        existing_fact = existing_edge.fact.strip()
        if new_fact and new_fact not in existing_fact and existing_fact not in new_fact:
            existing_edge.fact = f'{existing_fact} {new_fact}'
            existing_edge.fact_embedding = None
            logger.info(f'[edge_dedup] Concatenated fact for edge {existing_edge.uuid}')
    # 合并属性、valid_at、invalid_at（同原逻辑）
    ...
```

### 合并策略

| 字段 | 合并策略 |
|-----|---------|
| `fact` | 优先使用 LLM 合并结果；如无则拼接（去重后） |
| `fact_embedding` | 清空，由后续流程重新计算 |
| `episodes` | 追加新 episode UUID（原有逻辑） |
| `attributes` | 新边补充旧边缺失的字段 |
| `valid_at` | 取较早的时间戳 |
| `invalid_at` | 取较晚的时间戳 |

### 修改文件清单

| 文件 | 修改内容 |
|-----|---------|
| `graphiti_core/prompts/dedupe_edges.py` | `EdgeDuplicate` 添加 `merged_fact` 字段 |
| `graphiti_core/prompts/dedupe_edges.py` | `resolve_edge` 提示词增加合并任务 |
| `graphiti_core/utils/maintenance/edge_operations.py` | `resolve_extracted_edge` 强制合并同一对节点的边 |
| `graphiti_core/utils/maintenance/edge_operations.py` | `resolve_extracted_edge` 支持拼接 fact（无 LLM 合并时） |

---

## 7. 实体去重 Reasoning 字段与 Prompt 优化

### 问题背景

实体去重时 LLM 做出错误判断（将不同实体判定为重复），但没有解释原因，难以调试和定位问题。

**具体案例**：`监控套件` 被错误添加了同义词 `SNMP监控套件`、`业务墙`、`拨测详情`。

### 解决方案

#### 8.1 添加 reasoning 字段

**文件**: `graphiti_core/prompts/dedupe_nodes.py`

在 `NodeDuplicate` 模型中添加 `reasoning` 字段：

```python
class NodeDuplicate(BaseModel):
    id: int = Field(...)
    duplicate_idx: int = Field(...)
    name: str = Field(...)
    duplicates: list[int] = Field(...)
    reasoning: str = Field(
        default='',
        description='Brief explanation of why this entity is or is not a duplicate. Required when duplicate_idx != -1.',
    )
```

#### 8.2 更新 Prompt 要求输出 reasoning

**文件**: `graphiti_core/prompts/dedupe_nodes.py`

修改 `nodes()` 和 `node()` 函数的 prompt，在输出格式中添加 reasoning 字段：

```python
For every entity, return an object with the following keys:
{{
    "id": integer id from ENTITIES,
    "name": the best full name for the entity,
    "duplicate_idx": the idx of the EXISTING ENTITY that is the best duplicate match, or -1 if there is no duplicate,
    "duplicates": a sorted list of all idx values from EXISTING ENTITIES that refer to duplicates,
    "reasoning": a brief explanation (1-2 sentences) of why you determined this entity is or is not a duplicate. REQUIRED when duplicate_idx != -1.
}}

- When marking as duplicate, explain what evidence shows they refer to the same real-world object.
- When NOT marking as duplicate, you may leave reasoning empty or briefly explain why they are distinct.
```

#### 8.3 优化 entity_type_definitions（避免重复）

**问题**：每个实体的 `entity_type_description` 完全重复，浪费 token。

**解决方案**：将 entity_type_definitions 独立为一个部分，在实体列表中只保留类型名称。

**文件**: `graphiti_core/utils/maintenance/node_operations.py`

```python
# 修改前：每个实体都带 entity_type_description
extracted_nodes_context = [
    {
        'id': i,
        'name': node.name,
        'entity_type': node.labels,
        'entity_type_description': entity_types_dict.get(...).__doc__,  # 重复！
    }
    for i, node in enumerate(llm_extracted_nodes)
]

# 修改后：独立提取类型定义
entity_type_definitions: dict[str, str] = {}
for node in llm_extracted_nodes:
    for label in node.labels:
        if label != 'Entity' and label not in entity_type_definitions:
            type_model = entity_types_dict.get(label)
            if type_model and type_model.__doc__:
                entity_type_definitions[label] = type_model.__doc__

extracted_nodes_context = [
    {
        'id': i,
        'name': node.name,
        'entity_type': node.labels,  # 只有类型名称，不重复描述
    }
    for i, node in enumerate(llm_extracted_nodes)
]

context = {
    'extracted_nodes': extracted_nodes_context,
    'existing_nodes': existing_nodes_context,
    'entity_type_definitions': entity_type_definitions,  # 独立部分
    ...
}
```

**文件**: `graphiti_core/prompts/dedupe_nodes.py`

修改 `nodes()` 函数，添加 `ENTITY TYPE DEFINITIONS` 部分：

```python
def nodes(context: dict[str, Any]) -> list[Message]:
    # Build entity type definitions section if available
    type_defs = context.get('entity_type_definitions', {})
    type_defs_section = ''
    if type_defs:
        type_defs_section = f"""
        <ENTITY TYPE DEFINITIONS>
        {to_prompt_json(type_defs)}
        </ENTITY TYPE DEFINITIONS>
        """

    return [
        Message(
            role='user',
            content=f"""
        ...
        {type_defs_section}

        Each entity in ENTITIES is represented as a JSON object with the following structure:
        {{
            id: integer id of the entity,
            name: "name of the entity",
            entity_type: ["Entity", "<optional additional label>", ...]
        }}
        ...
        """
        ),
    ]
```

#### 8.4 添加日志记录

**文件**: `graphiti_core/utils/maintenance/node_operations.py`

在 `_resolve_with_llm()` 函数中，判定为重复时记录 reasoning：

```python
if resolved_node.uuid != extracted_node.uuid:
    state.duplicate_pairs.append((extracted_node, resolved_node))
    # Log deduplication decision with reasoning for debugging
    logger.info(
        'Dedupe: "%s" -> "%s" (reasoning: %s)',
        extracted_node.name,
        resolved_node.name,
        resolution.reasoning or 'no reasoning provided',
    )
```

### 效果

1. **调试能力增强**：可以从日志中看到 LLM 判定重复的原因
2. **Token 节省**：entity_type_definitions 不再重复，每种类型只出现一次
3. **Prompt 结构清晰**：类型定义独立，实体列表更简洁

### 修改文件清单

| 文件 | 修改内容 |
|-----|---------|
| `graphiti_core/prompts/dedupe_nodes.py` | `NodeDuplicate` 添加 `reasoning` 字段 |
| `graphiti_core/prompts/dedupe_nodes.py` | `nodes()` 和 `node()` prompt 要求输出 reasoning |
| `graphiti_core/prompts/dedupe_nodes.py` | `nodes()` 添加 `ENTITY TYPE DEFINITIONS` 独立部分 |
| `graphiti_core/utils/maintenance/node_operations.py` | `_resolve_with_llm()` 独立提取 entity_type_definitions |
| `graphiti_core/utils/maintenance/node_operations.py` | `_resolve_with_llm()` 添加 reasoning 日志 |

---

## 8. 属性抽取枚举验证容错（Attribute Extraction Graceful Handling）

### 问题背景

LLM 抽取实体属性时，可能返回不在 Schema 枚举值中的字段值，导致 Pydantic 验证失败，整个批次导入崩溃。

**具体案例**：

```
ValidationError: 1 validation error for Component
component_type
  Input should be 'backend', 'middleware', 'database', 'frontend', 'script', 'storage' or 'agent'
  [type=literal_error, input_value='feature', input_type=str]
```

LLM 把 `component_type` 填成了 `'feature'`，但 Schema 只允许特定的枚举值。

### 解决方案

**文件**: `graphiti_core/utils/maintenance/node_operations.py`

修改 `_extract_entity_attributes()` 函数，添加 `ValidationError` 的容错处理：

1. 捕获 `ValidationError` 而不是直接抛出
2. 从错误信息中提取验证失败的字段名
3. 从响应中移除这些无效字段
4. 返回清理后的响应

```python
from pydantic import BaseModel, ValidationError

async def _extract_entity_attributes(...) -> dict[str, Any]:
    # ... 省略 ...

    # validate response with graceful error handling for invalid enum values
    try:
        entity_type(**llm_response)
    except ValidationError as e:
        # EasyOps customization: handle invalid enum values gracefully
        logger.warning(f'Entity attribute validation warning: {e}. Will remove invalid fields.')

        # Extract field names that have validation errors
        invalid_fields = set()
        for error in e.errors():
            if error.get('loc'):
                field_name = error['loc'][0]
                invalid_fields.add(field_name)
                logger.warning(f'Removing invalid field "{field_name}"')

        # Remove invalid fields and try again
        cleaned_response = {k: v for k, v in llm_response.items() if k not in invalid_fields}

        # Validate cleaned response
        try:
            entity_type(**cleaned_response)
            return cleaned_response
        except ValidationError as e2:
            logger.error(f'Validation still failed after cleanup: {e2}')
            return {}  # Return empty dict rather than crash

    return llm_response
```

### 效果

- 当 LLM 返回无效枚举值时，只丢弃该字段，保留其他有效属性
- 批量导入不会因为单个字段验证失败而崩溃
- 日志记录被移除的字段，便于后续优化 Schema 或 Prompt

### 修改文件清单

| 文件 | 修改内容 |
|-----|---------|
| `graphiti_core/utils/maintenance/node_operations.py` | 添加 `ValidationError` import |
| `graphiti_core/utils/maintenance/node_operations.py` | `_extract_entity_attributes()` 添加枚举验证容错逻辑 |
| `graphiti_core/prompts/extract_nodes.py` | `filter_entities()` 增加 Schema 实体类型上下文 |
| `graphiti_core/utils/maintenance/node_operations.py` | `filter_extracted_nodes()` 传入 entity_types_context |

---

## 9. Filter Entities 增加 Schema 上下文（Schema-Aware Filtering）

### 问题背景

`filter_entities` 步骤原本只应用四大原则过滤实体，但没有领域 Schema 知识：

1. **误删合法实体**：`IT服务中心`（ProductModule）被误删，因为 LLM 认为它是"UI navigation"
2. **误删功能名称**：`服务目录`（Feature）被误删，因为 LLM 认为它是"document-specific section title"

**根因**：filter 步骤没有拿到 Schema 实体类型定义，不知道哪些是合法的领域概念。

### 解决方案

让 filter 步骤拿到完整的 Schema 实体类型定义，在应用四大原则前先检查实体是否匹配已定义的类型。

#### 9.1 修改 filter_entities 提示词

**文件**: `graphiti_core/prompts/extract_nodes.py`

```python
def filter_entities(context: dict[str, Any]) -> list[Message]:
    sys_prompt = """You are a knowledge graph quality reviewer...

**IMPORTANT**: Before removing an entity, check if it matches any of the VALID ENTITY TYPES defined in the schema. If an entity clearly belongs to a defined type (based on the type's description and examples), it should be KEPT even if it seems document-specific."""

    # Build entity types reference if available
    entity_types_ref = ''
    if context.get('entity_types'):
        entity_types_ref = '\n<VALID ENTITY TYPES>\n'
        for et in context['entity_types']:
            if et.get('entity_type_name') != 'Entity':  # Skip default type
                entity_types_ref += f"- {et.get('entity_type_name')}: {et.get('entity_type_description', '')}\n"
        entity_types_ref += '</VALID ENTITY TYPES>\n'

    user_prompt = f"""
...
{entity_types_ref}
Review each extracted entity. Return the names of entities that FAIL the Knowledge Graph Builder's Principles.

**Decision Process**:
1. First, check if the entity matches any VALID ENTITY TYPE (if provided). If it clearly fits a defined type based on the type's description and examples, KEEP it.
2. Only if the entity doesn't match any valid type, apply the four principles strictly.
...
"""
```

#### 9.2 修改 filter_extracted_nodes 函数

**文件**: `graphiti_core/utils/maintenance/node_operations.py`

添加 `entity_types_context` 参数：

```python
async def filter_extracted_nodes(
    llm_client: LLMClient,
    episode: EpisodicNode,
    extracted_entities: list[ExtractedEntity],
    group_id: str | None = None,
    entity_types_context: list[dict] | None = None,  # 新增
) -> list[str]:
    """Filter out entities that don't meet knowledge graph quality standards.

    If entity_types_context is provided, entities matching valid types will be preserved.
    """
    if not extracted_entities:
        return []

    context = {
        'episode_content': episode.content,
        'extracted_entities': [e.name for e in extracted_entities],
        'entity_types': entity_types_context,  # 传入 Schema 类型定义
    }
    # ...
```

#### 9.3 修改调用方

**文件**: `graphiti_core/utils/maintenance/node_operations.py`

在 `extract_nodes()` 函数中传入 `entity_types_context`：

```python
# Filter entities using Knowledge Graph Builder's Principles
entities_to_remove = await filter_extracted_nodes(
    llm_client,
    episode,
    extracted_entities,
    episode.group_id,
    entity_types_context,  # 新增：传入实体类型上下文
)
```

### 决策流程

修改后的 filter 决策流程：

```
对每个待过滤的实体：
  1. 检查是否匹配 Schema 定义的类型
     - 匹配 → KEEP（即使看起来像文档元素）
     - 不匹配 → 进入下一步
  2. 应用四大原则
     - 通过全部 → KEEP
     - 任一失败 → REMOVE
```

### 效果

| 实体 | 修改前 | 修改后 | 原因 |
|-----|-------|-------|------|
| IT服务中心 | ❌ 误删 | ✅ 保留 | 匹配 ProductModule 定义 |
| 服务目录 | ❌ 误删 | ✅ 保留 | 匹配 Feature 定义 |
| 流程编排 | ❌ 误删 | ✅ 保留 | 匹配 Feature 定义 |
| 测试验证 | ✅ 过滤 | ✅ 过滤 | 不匹配任何类型 + 违反独立性 |
| 运维审批 | ✅ 过滤 | ✅ 过滤 | 不匹配任何类型 + 违反独立性 |

### 修改文件清单

| 文件 | 修改内容 |
|-----|---------|
| `graphiti_core/prompts/extract_nodes.py` | `filter_entities()` 增加 entity_types 上下文和决策流程 |
| `graphiti_core/utils/maintenance/node_operations.py` | `filter_extracted_nodes()` 新增 `entity_types_context` 参数 |
| `graphiti_core/utils/maintenance/node_operations.py` | `extract_nodes()` 调用时传入 `entity_types_context` |

---

## 10. 批量语义去重性能优化（Semantic Dedup Batch Optimization）- **已回滚**

### 问题背景

`add_episode_bulk` 批量导入时，`semantic_dedupe_nodes_bulk` 函数对同类型实体进行 O(n²) 的串行 LLM 调用：

```python
# 原实现：54 个 Feature 实体 → 54 次串行 LLM 调用
for i, node in enumerate(type_nodes):
    candidates = [n for n in type_nodes[i+1:] if n.uuid not in duplicate_map]
    llm_pairs = await _resolve_batch_with_llm([node], candidates, entity_types)
```

性能分析显示：54 个 Feature 实体的语义去重耗时 214 秒（占总时间 42%）。

### 曾经的优化方案（已回滚）

曾将 O(n²) 串行调用改为 O(entity_type_count) 批量调用，每种实体类型只需 1 次 LLM 调用。

### 回滚原因（2025-12-15）

批量自比较的实现有逻辑缺陷：把所有节点同时作为 ENTITIES 和 EXISTING ENTITIES，导致 LLM 每个节点都能在 EXISTING 中找到自己（100% 匹配），从而不会去比较其他节点是否重复。

**具体表现**：`ITSC管理平台` 和 `EasyITSM` 虽然 `code="ITSC"` 相同且 summary 功能描述高度相似，但 LLM 各自匹配到自己，没有识别为重复。

### 当前实现

已回滚到 O(n²) 实现，确保准确性优先：

```python
async def _resolve_batch_with_llm(
    llm_client: LLMClient,
    nodes: list[EntityNode],        # 待检查的节点
    candidates: list[EntityNode],   # 候选节点（不包含自己）
    entity_types: dict[str, type[BaseModel]] | None,
) -> list[tuple[EntityNode, EntityNode]]:
    """Compare nodes against candidates (not including self) to find duplicates."""
```

```python
async def semantic_dedupe_nodes_bulk(...):
    for i, node in enumerate(type_nodes):
        candidates = [n for n in type_nodes[i+1:] if n.uuid not in duplicate_map]
        llm_pairs = await _resolve_batch_with_llm(
            clients.llm_client, [node], candidates, entity_types
        )
```

### 修改文件清单

| 文件 | 修改内容 |
|-----|---------|
| `graphiti_core/utils/bulk_utils.py` | `_resolve_batch_self_dedup()` → `_resolve_batch_with_llm()` |
| `graphiti_core/utils/bulk_utils.py` | `semantic_dedupe_nodes_bulk()` 回滚到 O(n²) 实现 |

---

## 11. 实体去重 Prompt 强化（Dedupe Prompt Strictness）- **已回滚**

### 问题背景

原有去重 prompt 导致 LLM 将相关但不同的实体错误合并：
- 不同操作被合并（如 "Approve" vs "Reject"）
- 同类别不同功能被合并（如 "User Management" vs "Role Management"）

### 解决方案（已回滚）

**文件**: `graphiti_core/prompts/dedupe_nodes.py`

曾在 `nodes()` 函数中增加严格的判断标准：

```python
## STRICT DUPLICATE CRITERIA

Entities are duplicates ONLY if they refer to **the EXACT same real-world object or concept**.

**TRUE DUPLICATES** (should merge):
- Same entity, different names (e.g., abbreviation vs full name, or translation)
- Typos or minor spelling variations of the same entity
- Synonyms that refer to the identical concept in this domain

**NOT DUPLICATES** (keep separate):
- Different operations/actions in the same domain
- Different features in the same category
- Related but conceptually distinct items
- Parent-child or hierarchical relationships
- Items with similar names but different purposes or scopes

## DECISION RULE

When uncertain, **DO NOT merge**. It is better to have two separate entities
than to incorrectly merge distinct concepts.

Ask yourself: "Are these two names referring to the IDENTICAL thing,
or are they two different things that happen to be related?"
```

### 回滚原因（2025-12-12）

上述修改太过保守，导致 LLM 不合并明显应该合并的实体：
- EasyITSM 和 EasyITSC 不被合并，尽管它们的 `module_name="EasyITSC"` 和 `code="ITSC"` 完全一致
- LLM 看到名字不同就"uncertain"，然后因为 "When uncertain, DO NOT merge" 而不合并

**已恢复到 Graphiti 原始 prompt**：

```python
Entities should only be considered duplicates if they refer to the *same real-world object or concept*.

Do NOT mark entities as duplicates if:
- They are related but distinct.
- They have similar names or purposes but refer to separate instances or concepts.
```

原始 prompt 更简洁，没有"When uncertain, DO NOT merge"这种极端保守的规则，LLM 可以正确利用 attributes（如 module_name、code）来判断重复。

### 修改文件清单

| 文件 | 修改内容 |
|-----|---------|
| `graphiti_core/prompts/dedupe_nodes.py` | ~~`nodes()` 增加 STRICT DUPLICATE CRITERIA 和 DECISION RULE~~ **已回滚** |

---

## 12. 去重优先级明确（Deduplication Priority）

### 问题背景

实体去重时，需要明确 name、summary 和 attributes（如 code）的优先级关系。

**错误做法**：让 LLM 过度依赖 code 属性，导致可能将 code 相同但实际不同的实体错误合并。

**正确做法**：name 和 summary 是主要判断依据，attributes 是辅助证据。

### 解决方案

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

### 关键原则

| 优先级 | 证据 | 说明 |
|-------|-----|------|
| 1 | Name + Summary | 主要依据，如果 summary 明确描述同一概念则为重复 |
| 2 | Attributes | 辅助证据，支持去重判断但不作为唯一依据 |
| 3 | 禁止单独依赖 attributes | 如果 name/summary 描述不同概念，即使 code 相同也不应合并 |

### 修改文件清单

| 文件 | 修改内容 |
|-----|---------|
| `graphiti_core/prompts/dedupe_nodes.py` | `node()` 和 `nodes()` 明确 Deduplication Priority |

---

## 13. Filter Entities 类型重新验证与重新分类（Type Re-validation and Reclassification）

### 问题背景

`filter_entities` 步骤在检查实体是否应该保留时，会检查实体是否匹配 VALID ENTITY TYPES。但存在一个漏洞：

1. 实体在抽取阶段可能被错误分类（如监控指标 `system_cpu_cores` 被标记为 `Component`）
2. Filter 步骤看到实体有 `type: "Component"` 标签
3. LLM 看到 `Component` 是有效类型，就直接保留了
4. **没有验证实体是否真正符合 Component 的定义**

**具体案例**（trace_id: d1d3dcc90794422ee5c301af96fd6c62）：

```json
// 抽取阶段错误分类
{
  "name": "system_socket_summary_udp_all_count",
  "summary": "...",
  "type": "Component"  // 错误！这是监控指标，不是组件
}

// Filter 阶段 LLM 的错误推理
{
  "entities_to_remove": [],
  "reasoning": "All extracted entities represent system monitoring metrics
   that are core components of the EasyOps monitoring system..."
}
```

Component 的定义要求：
> 有独立的进程/服务/安装包？运维可以独立启停、配置、监控？

但 `system_socket_summary_udp_all_count` 是一个**监控指标 ID**，不符合 Component 定义。

### 解决方案

#### 13.1 新增类型重新分类功能

**原有设计**：误分类的实体只能被删除，可能丢失有价值的实体。

**新设计**：误分类的实体可以被重新分类到正确的类型，保留有价值的实体。

**新增模型** (`graphiti_core/prompts/extract_nodes.py`)：

```python
class EntityReclassification(BaseModel):
    name: str = Field(..., description='Name of the entity to reclassify')
    new_type: str = Field(
        ...,
        description='The correct entity type name from VALID ENTITY TYPES. Must be an exact match.',
    )
    reason: str = Field(..., description='Brief explanation of why this type is more appropriate')


class EntitiesToFilter(BaseModel):
    entities_to_remove: list[str] = Field(...)
    entities_to_reclassify: list[EntityReclassification] = Field(  # 新增
        default_factory=list,
        description='Entities that were misclassified and should be assigned a different type',
    )
    reasoning: str = Field(...)
```

#### 13.2 修改 Filter 提示词

**System Prompt 增加 CRITICAL 提示**：

```python
sys_prompt = """You are a knowledge graph quality reviewer...

**CRITICAL**: Entities may have a pre-assigned "type" from the extraction step.
Do NOT trust this type blindly. You MUST re-validate the entity against the
type's definition in VALID ENTITY TYPES. If the entity does not actually match
its assigned type's criteria (especially the IS/IS NOT examples in the type description):
- If it matches a DIFFERENT valid type, RECLASSIFY it
- If it doesn't match ANY valid type, REMOVE it"""
```

**Decision Process 改为类型重新验证 + 重新分类**：

```python
**Decision Process**:
1. If the entity has a pre-assigned type, RE-VALIDATE it against that type's definition:
   - Check if it matches the type's criteria (judgment standards)
   - Check if it matches the IS examples (should be kept with current type)
   - Check if it matches the IS NOT examples (needs reclassification or removal)
   - If MISCLASSIFIED but matches a DIFFERENT valid type → add to entities_to_reclassify
   - If MISCLASSIFIED and doesn't match ANY valid type → add to entities_to_remove
2. If the entity has no type or type is "Entity", check if it matches any valid type:
   - If it matches a valid type → add to entities_to_reclassify with the correct type
   - If it doesn't match any type, apply the four principles strictly
```

**添加常见误分类提示**：

```python
**Common Misclassifications to Watch For**:
- Metric IDs (like system_cpu_cores, system_memory_total) are NOT Components - Components are deployable services
- Table columns or field names are NOT CmdbModels - CmdbModels are model definitions
- Generic technical terms are NOT Features - Features have menu entries in the product
```

#### 13.3 修改 filter_extracted_nodes 函数

**文件**: `graphiti_core/utils/maintenance/node_operations.py`

返回类型从 `list[str]` 改为 `tuple[list[str], list[EntityReclassification]]`：

```python
async def filter_extracted_nodes(
    llm_client: LLMClient,
    episode: EpisodicNode,
    nodes: list[EntityNode],
    group_id: str | None = None,
    entity_types_context: list[dict] | None = None,
) -> tuple[list[str], list[EntityReclassification]]:
    """Filter out entities and identify misclassified entities for reclassification.

    Returns:
        Tuple of (entities_to_remove, entities_to_reclassify)
    """
```

#### 13.4 修改调用方处理重新分类

**文件**: `graphiti_core/graphiti.py`

**单个 Episode 处理**（`add_episode` 方法）：

```python
entities_to_remove, entities_to_reclassify = await filter_extracted_nodes(...)

# Apply reclassifications to nodes
if entities_to_reclassify:
    reclassify_map = {r.name: r.new_type for r in entities_to_reclassify}
    for node in hydrated_nodes:
        if node.name in reclassify_map:
            new_type = reclassify_map[node.name]
            old_labels = node.labels.copy()
            node.labels = ['Entity', new_type] if new_type != 'Entity' else ['Entity']
            logger.info(f'Reclassified "{node.name}": {old_labels} -> {node.labels}')
```

**批量 Episode 处理**（`_resolve_nodes_and_edges_bulk` 方法）：

```python
# Collect all reclassifications
all_reclassifications: dict[str, str] = {}  # name -> new_type
for entities_to_remove, entities_to_reclassify in filter_results:
    for reclass in entities_to_reclassify:
        all_reclassifications[reclass.name] = reclass.new_type

# Apply reclassifications
if all_reclassifications:
    for nodes in hydrated_nodes_results:
        for node in nodes:
            if node.name in all_reclassifications:
                new_type = all_reclassifications[node.name]
                node.labels = ['Entity', new_type] if new_type != 'Entity' else ['Entity']
```

### 效果

| 实体 | 修改前 | 修改后 | 原因 |
|-----|-------|-------|------|
| `system_cpu_cores` (误标为 Component) | ✅ 保留（误） | ❌ 移除 | 不符合任何有效类型 |
| `IT服务中心` (误标为 Entity) | ✅ 保留但类型错误 | ✅ 重新分类为 ProductModule | 符合 ProductModule 定义 |
| `cmdb_service` | ✅ 保留 | ✅ 保留 | 符合 Component 定义（独立服务） |

### 修改文件清单

| 文件 | 修改内容 |
|-----|---------|
| `graphiti_core/prompts/extract_nodes.py` | 新增 `EntityReclassification` 模型 |
| `graphiti_core/prompts/extract_nodes.py` | `EntitiesToFilter` 添加 `entities_to_reclassify` 字段 |
| `graphiti_core/prompts/extract_nodes.py` | `filter_entities()` 提示词支持重新分类 |
| `graphiti_core/utils/maintenance/node_operations.py` | `filter_extracted_nodes()` 返回类型改为 tuple |
| `graphiti_core/graphiti.py` | `add_episode()` 处理重新分类 |
| `graphiti_core/graphiti.py` | `_resolve_nodes_and_edges_bulk()` 处理重新分类 |

---

## 14. 两步类型验证优化（Two-Step Type Validation）

### 问题背景

原有的 `filter_entities` 提示词一次性传入：
- 所有 17+ 个 Schema 类型定义
- 长篇的 episode 原文内容
- 所有待验证的实体列表

这导致 LLM 注意力分散，无法准确判断每个实体是否符合其分配的类型定义。

**具体表现**：监控指标 ID（如 `system_cpu_cores`）被错误标记为 `Component`，但 LLM 在 filter 阶段看到 `Component` 是有效类型就直接保留了，没有验证该实体是否真正符合 Component 的定义（独立进程/服务/安装包）。

### 解决方案

将原来的单次 filter 调用拆分为两步：

**Step 1: 聚焦验证（Focused Validation）**
- 每个实体只传入**自己的**类型定义
- 不传入 episode 原文（只用 summary）
- 批量处理（5 个实体/批），并行执行
- 判断：实体是否符合其分配的类型定义？

**Step 2: 重新分类（Reclassification）**
- 只对 Step 1 中验证失败的实体
- 传入**所有**类型定义
- 判断：是否有其他类型匹配？还是应该删除？

### 实现细节

#### 14.1 新增数据模型

**文件**: `graphiti_core/prompts/extract_nodes.py`

```python
# Step 1: 验证结果
class EntityValidationItem(BaseModel):
    name: str = Field(..., description='Name of the entity')
    is_valid: bool = Field(..., description='True if entity matches its assigned type definition')
    reason: str = Field(..., description='Brief explanation of why it matches or does not match')

class EntityValidationResult(BaseModel):
    validations: list[EntityValidationItem] = Field(..., description='Validation results for each entity')

# Step 2: 重新分类结果
class EntityReclassifyItem(BaseModel):
    name: str = Field(..., description='Name of the entity')
    new_type: str | None = Field(..., description='New type name if it matches another type, or null if should be removed')
    reason: str = Field(..., description='Brief explanation of the decision')

class EntityValidationResult(BaseModel):
    validations: list[EntityValidationItem] = Field(..., description='Validation results for each entity')
```

#### 14.2 提示词函数

**Step 1 - `validate_entity_types(context)`**（新增）：
- 输入：实体列表，每个实体包含 `name`、`summary`、`assigned_type`、`type_definition`
- 不传入 episode 原文，只有实体自身信息
- 验证规则：检查 IS/IS NOT 示例，保守判断

**Step 2 - 复用 `extract_text(context)`**（生产验证过的提示词）：
- 输入：实体的 name+summary 作为"文本"，所有类型定义
- 让 LLM 重新抽取分类
- 如果分类为 Entity 或原类型 → 删除；否则重新分类

#### 14.3 重写 filter_extracted_nodes

**文件**: `graphiti_core/utils/maintenance/node_operations.py`

```python
async def filter_extracted_nodes(...) -> tuple[list[str], list[EntityReclassification]]:
    # Step 1: Batch validation (5 entities per batch, parallel)
    async def validate_batch(batch):
        context = {'entities': batch}  # 每个实体只有自己的 type_definition
        return await llm_client.generate_response(
            prompt_library.extract_nodes.validate_entity_types(context),
            EntityValidationResult,
        )

    validation_results = await semaphore_gather(*[validate_batch(b) for b in batches])

    # Collect invalid entities
    invalid_entities = [e for e in validation_results if not e.is_valid]

    if not invalid_entities:
        return [], []

    # Step 2: Reclassify using production-validated extract_text prompt (parallel)
    async def reclassify_entity(entity):
        entity_text = f"Entity: {entity['name']}\nDescription: {entity['summary']}"
        context = {
            'episode_content': entity_text,
            'entity_types': entity_types_context,  # 所有类型定义
            # NOTE: Do NOT include validation_reason here - it may contain type suggestions
            # that would bias the LLM. Let extract_text make an independent classification.
            'custom_prompt': f"Classify the entity '{entity['name']}' based on its description. "
                           f"The entity was previously classified as '{entity['assigned_type']}' "
                           f"but that classification was incorrect. "
                           f"Please determine the correct entity type from the available types.",
        }
        return await llm_client.generate_response(
            prompt_library.extract_nodes.extract_text(context),  # 复用生产验证的提示词
            ExtractedEntities,
        )

    reclassify_results = await semaphore_gather(*[reclassify_entity(e) for e in invalid_entities])
    # ...
```

**重要**：Step 2 的 `custom_prompt` 不包含 Step 1 的 `validation_reason`，避免 LLM 被前一步的建议所偏向。

#### 14.4 重新分类后更新 node.reasoning

**文件**: `graphiti_core/graphiti.py`

重新分类后不仅更新 `node.labels`，还要更新 `node.reasoning`，避免保留原类型的推理内容：

```python
# Apply reclassifications to nodes
if entities_to_reclassify:
    # Store both new_type and reason for updating node.reasoning
    reclassify_map = {r.name: (r.new_type, r.reason) for r in entities_to_reclassify}
    for node in hydrated_nodes:
        if node.name in reclassify_map:
            new_type, reason = reclassify_map[node.name]
            old_labels = node.labels.copy()
            node.labels = ['Entity', new_type] if new_type != 'Entity' else ['Entity']
            # Update reasoning with reclassification reason
            node.reasoning = f'[Reclassified from {old_labels} to {node.labels}] {reason}'
            logger.info(f'Reclassified "{node.name}": {old_labels} -> {node.labels}, reason: {reason}')
```

### 性能对比

| 指标 | 原方案 | 两步方案 |
|-----|-------|---------|
| LLM 调用次数 | 1 次（所有实体） | N/5 次（Step 1）+ M 次（Step 2，M=失败数） |
| 单次上下文大小 | 大（17 类型 + episode 原文） | 小（Step 1: 5实体×1类型; Step 2: 1实体×17类型） |
| 并行度 | 无 | **两步都并行** |
| 准确率 | 低（注意力分散） | 高（聚焦验证 + 复用生产验证提示词） |

### 效果

| 实体 | 原方案 | 两步方案 | 原因 |
|-----|-------|---------|------|
| `system_cpu_cores` (误标为 Component) | ✅ 保留（误） | ❌ Step 1 失败 → Step 2 无匹配 → 删除 | 不符合 Component 定义 |
| `监控套件` (正确标为 Feature) | ✅ 保留 | ✅ Step 1 通过 | 符合 Feature 定义 |
| `IT服务中心` (误标为 Entity) | ✅ 保留 | ✅ Step 1 失败 → Step 2 重分类为 ProductModule | 符合 ProductModule 定义 |

### 修改文件清单

| 文件 | 修改内容 |
|-----|---------|
| `graphiti_core/prompts/extract_nodes.py` | 新增 `EntityValidationItem`、`EntityValidationResult` 模型 |
| `graphiti_core/prompts/extract_nodes.py` | 新增 `validate_entity_types()` 提示词函数（Step 1） |
| `graphiti_core/prompts/extract_nodes.py` | 更新 `Versions` TypedDict 和 `versions` dict |
| `graphiti_core/utils/maintenance/node_operations.py` | 重写 `filter_extracted_nodes()` 为两步逻辑 |
| `graphiti_core/utils/maintenance/node_operations.py` | Step 2 的 `custom_prompt` 不包含 `validation_reason`（避免偏向） |
| `graphiti_core/utils/maintenance/node_operations.py` | 添加批量并行处理（5 实体/批）|
| `graphiti_core/graphiti.py` | 重新分类时同时更新 `node.reasoning`（`add_episode` 和 `_resolve_nodes_and_edges_bulk`） |

---

## 15. 重新分类时清理旧类型属性（Attribute Cleanup on Reclassification）

### 问题背景

当实体从一个类型重新分类到另一个类型时，原有代码只更新了 `node.labels` 和 `node.reasoning`，但没有清理 `node.attributes` 中属于旧类型的属性。

**具体案例**：

```
1. 实体 "流程库" 初始被错误分类为 ProductModule
2. 属性抽取阶段提取了 ProductModule 属性：module_name="EasyITSC", code="ITSC"
3. 两步验证检测到误分类，重新分类为 Feature
4. node.labels 更新为 ['Entity', 'Feature']
5. 但 node.attributes 仍然包含 module_name, code（属于 ProductModule）
6. 这些污染属性被写入数据库并通过去重传播
```

**最终结果**：数据库中 Feature 实体包含 ProductModule 的属性（属性污染）。

### 解决方案

**文件**: `graphiti_core/graphiti.py`

**位置**: `add_episode()` 方法和 `_resolve_nodes_and_edges_bulk()` 方法中的重分类逻辑

在更新 `node.labels` 后，根据新类型的 Schema 清理不属于新类型的属性：

```python
# Apply reclassifications to nodes
if entities_to_reclassify:
    reclassify_map = {r.name: (r.new_type, r.reason) for r in entities_to_reclassify}
    for node in hydrated_nodes:
        if node.name in reclassify_map:
            new_type, reason = reclassify_map[node.name]
            old_labels = node.labels.copy()
            node.labels = ['Entity', new_type] if new_type != 'Entity' else ['Entity']

            # Clear attributes not belonging to new type schema
            # This prevents attribute pollution when reclassifying from one type to another
            if entity_types and new_type in entity_types:
                new_type_model = entity_types[new_type]
                valid_fields = set(new_type_model.model_fields.keys())
                old_attrs = node.attributes.copy()
                node.attributes = {k: v for k, v in node.attributes.items() if k in valid_fields}
                if old_attrs != node.attributes:
                    removed_attrs = set(old_attrs.keys()) - set(node.attributes.keys())
                    logger.info(f'Cleared invalid attributes from "{node.name}": {removed_attrs}')
            elif new_type == 'Entity':
                # Reclassified to generic Entity, clear all custom attributes
                if node.attributes:
                    logger.info(f'Cleared all attributes from "{node.name}" (reclassified to Entity)')
                    node.attributes = {}

            # Update reasoning with reclassification reason
            node.reasoning = f'[Reclassified from {old_labels} to {node.labels}] {reason}'
```

### 清理逻辑

| 新类型 | 清理策略 |
|-------|---------|
| 有 Schema 定义的类型（如 Feature） | 只保留新类型 Schema 中定义的字段 |
| 通用 Entity 类型 | 清除所有自定义属性 |

### 日志输出

- `Cleared invalid attributes from "流程库": {'module_name', 'code'}`
- `Cleared all attributes from "xxx" (reclassified to Entity)`

### 效果

| 场景 | 修复前 | 修复后 |
|-----|-------|-------|
| ProductModule → Feature | attributes 保留 module_name, code | 只保留 feature_name, description 等 |
| Feature → Entity | attributes 保留 feature_name | attributes 清空 |
| Component → Feature | attributes 保留 component_type | 只保留 Feature Schema 字段 |

### 修改文件清单

| 文件 | 修改内容 |
|-----|---------|
| `graphiti_core/graphiti.py` | `add_episode()` 重分类后清理属性 |
| `graphiti_core/graphiti.py` | `_resolve_nodes_and_edges_bulk()` 批量重分类后清理属性 |

---

## 22. 批量保存补充 type_scores 和 type_confidence 字段

### 问题背景

LLM 抽取实体时使用 `extract_text_with_scores` 或 `extract_message_with_scores` 提示词，会返回每个实体类型的置信度分数。这些分数被正确存储到 `EntityNode` 对象中，但在批量保存时丢失。

问题链路：
1. `node_operations.py:459-460` - 正确设置 `type_scores` 和 `type_confidence` 到 EntityNode
2. `bulk_utils.py:463-472` - 构建 `entity_data` 时**遗漏**了这两个字段
3. 结果：数据库中 `type_scores` 和 `type_confidence` 始终为 null

### 解决方案

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

### 对比 nodes.py 的 save() 方法

`nodes.py:502-503` 中的实现作为参考：

```python
'type_scores': json.dumps(self.type_scores) if self.type_scores else None,
'type_confidence': self.type_confidence,
```

### 修改文件清单

| 文件 | 修改内容 |
|-----|---------|
| `graphiti_core/utils/bulk_utils.py` | `entity_data` 补充 `type_scores` 和 `type_confidence` 字段 |

---

## 23. 实体类型打分优化 - Top 3 候选 + 二次推理

### 问题背景

原有的实体类型打分机制要求 LLM 对**所有**实体类型进行打分，存在以下问题：

1. **Token 消耗过大**：如果有 10 个实体类型，每个实体需要输出 10 组 (type_id, score, reasoning)
2. **效率低下**：大部分类型的分数都很低，没有必要输出
3. **歧义处理不足**：当多个类型都有较高分数时，缺乏进一步判断机制

### 解决方案

采用**两阶段类型分类**策略：

#### 阶段 1：Top 3 候选打分

只输出最可能的 3 个类型候选，每个候选包含分数和推理说明。

**修改文件**：`graphiti_core/prompts/extract_nodes.py`

```python
class TopTypeCandidate(BaseModel):
    """Top candidate type with score and reasoning."""
    type_id: int
    score: float  # 0.0 - 1.0
    reasoning: str  # 解释为什么给这个分数

class ExtractedEntityWithScores(BaseModel):
    """Entity with top 3 candidate types."""
    name: str
    top_candidates: list[TopTypeCandidate]  # 最多 3 个，按 score 降序
    final_type_id: int  # 最高分候选的 type_id
```

**提示词修改**：`extract_text_with_scores()` 和 `extract_message_with_scores()`

```python
**TYPE SCORING PROCESS**:
For each entity you extract:
1. Quickly scan ALL entity types to identify the 3 most likely candidates
2. Score each of these 3 candidates from 0.0 to 1.0
3. Provide reasoning for each score
4. Set final_type_id to the highest-scoring candidate
```

#### 阶段 2：歧义消解（可选）

当多个候选的分数都 >= 0.7 时，触发二次推理：

1. 收集所有需要消解的实体
2. **批量**调用一次 LLM 进行最终分类
3. **不传入**第一轮的分数和推理，避免偏向

**新增模型**：

```python
class ResolvedEntityType(BaseModel):
    """Resolved type for an ambiguous entity."""
    name: str
    chosen_type_id: int
    reasoning: str  # 解释为什么选择这个类型

class ResolvedEntityTypes(BaseModel):
    """Batch resolution results for ambiguous entities."""
    resolutions: list[ResolvedEntityType]
```

**新增提示词**：`resolve_ambiguous_types()`

```python
def resolve_ambiguous_types(context: dict[str, Any]) -> list[Message]:
    """Resolve ambiguous entity types when multiple candidates >= 0.7.

    只提供候选类型定义，不传入第一轮的分数和推理，避免偏向。
    """
```

### 处理流程

```
Episode 写入
    │
    ▼
┌─────────────────────────────────────┐
│ 阶段 1: Top 3 候选打分               │
│ - 对每个实体输出最可能的 3 个类型      │
│ - 每个候选包含 score 和 reasoning    │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 检查歧义                             │
│ - 统计每个实体中 score >= 0.7 的候选数│
│ - 如果 > 1 且 Top1-Top2 差距 < 15%   │
│   → 标记为需要二次推理               │
│ - 如果差距 >= 15%，信任第一轮结果     │
└─────────────────────────────────────┘
    │
    ▼ (如果有歧义实体)
┌─────────────────────────────────────┐
│ 阶段 2: 批量歧义消解                  │
│ - 收集所有歧义实体                    │
│ - 一次 LLM 调用批量处理               │
│ - 只传候选类型定义，不传分数/推理      │
└─────────────────────────────────────┘
    │
    ▼
最终实体类型确定
```

### 性能优化效果

| 场景 | 优化前 | 优化后 |
|-----|-------|-------|
| 10 个实体类型 | 每实体输出 10 组打分 | 每实体输出 3 组打分 |
| Token 消耗 | ~300 tokens/实体 | ~100 tokens/实体 |
| 歧义实体 | 直接取最高分 | 仅分数接近时二次推理 |

### 阈值配置

| 阈值 | 值 | 用途 |
|-----|-----|------|
| `TYPE_CONFIDENCE_THRESHOLD` | 0.6 | 低于此分数降级为 Entity |
| `AMBIGUOUS_SCORE_THRESHOLD` | 0.7 | 高于此分数的候选参与歧义检查 |
| `AMBIGUOUS_SCORE_GAP_THRESHOLD` | 0.15 | Top1-Top2 差距 >= 此值则跳过二次推理 |

**二次推理触发条件**（2025-12-18 优化）：
- 原条件：`len(high_score_candidates) > 1`（只要有多个高分候选就触发）
- 新条件：`len(high_score_candidates) > 1 AND (top1_score - top2_score) < 0.15`
- **优化原因**：当 Top1 明显领先 Top2 时（如 90% vs 70%，差距 20%），第一轮结果已经可信，无需二次推理。二次推理可能因缺少分数信息而做出错误判断。

### 第二轮也要打分（2025-12-18 优化）

**问题**：原有的第二轮只需要"选一个"，没有打分约束。LLM 可能被某个类型的描述误导，做出比第一轮更差的判断。

**解决方案**：第二轮也采用与第一轮相同的打分机制，输出每个候选的分数和理由。

**新增模型**：
```python
class CandidateTypeScore(BaseModel):
    type_id: int
    score: float  # 0.0 - 1.0
    reasoning: str

class ResolvedEntityType(BaseModel):
    name: str
    chosen_type_id: int
    reasoning: str
    candidate_scores: list[CandidateTypeScore]  # 新增：每个候选的分数
```

**type_scores 存储格式**（保留两轮历史）：
```python
{
  "Feature": {
    "score": 0.85,           # 最终分数（第二轮）
    "reasoning": "...",      # 最终理由（第二轮）
    "pass1_score": 0.9,      # 第一轮分数
    "pass1_reasoning": "..." # 第一轮理由
  },
  "ProductModule": {
    "score": 0.75,
    "reasoning": "...",
    "pass1_score": 0.7,
    "pass1_reasoning": "..."
  }
}
```

**前端显示**：EntityDrawer 组件显示两轮分数，按分数降序排列，pass1 数据以灰色斜体显示。

### 修改文件清单

| 文件 | 修改内容 |
|-----|---------|
| `graphiti_core/prompts/extract_nodes.py` | `TopTypeCandidate` 模型（替代全量 `TypeScore`） |
| `graphiti_core/prompts/extract_nodes.py` | `ExtractedEntityWithScores` 改用 `top_candidates` |
| `graphiti_core/prompts/extract_nodes.py` | `CandidateTypeScore` 新增模型（第二轮打分） |
| `graphiti_core/prompts/extract_nodes.py` | `ResolvedEntityType` 新增 `candidate_scores` 字段 |
| `graphiti_core/prompts/extract_nodes.py` | `resolve_ambiguous_types()` 提示词增加打分要求 |
| `graphiti_core/prompts/extract_nodes.py` | `extract_text_with_scores()` Top 3 逻辑 |
| `graphiti_core/prompts/extract_nodes.py` | `extract_message_with_scores()` Top 3 逻辑 |
| `graphiti_core/utils/maintenance/node_operations.py` | 两阶段类型分类处理逻辑 |
| `graphiti_core/utils/maintenance/node_operations.py` | 第二轮保留 pass1 数据到 type_scores |
| `web/src/components/EntityDrawer/index.tsx` | 显示两轮分数（pass1_score, pass1_reasoning） |

---

## 20. 实体去重分批并行处理（LLM 输出截断修复）

### 问题背景

当一次导入的文档包含大量实体时（如 27 个），实体去重阶段需要将每个新实体与所有候选实体进行相似度评分。LLM 需要为每个新实体的每个候选输出 JSON 格式的评分数据，包含 `candidate_idx`, `similarity_score`, `is_same_entity`, `reasoning` 字段。

问题是：
- 27 个新实体 × 50+ 个候选 = 需要输出几千条评分数据
- 输出 JSON 达到 55000+ 字符
- 超过 LLM 的 `max_output_tokens` 限制，JSON 被截断
- 导致 `JSONDecodeError: Unterminated string` 错误

### 解决方案

将实体分批处理，每批并行调用 LLM：

1. **分批**：每批最多 5 个实体（`DEDUP_BATCH_SIZE = 5`）
2. **并行**：最多 10 个并发请求（`DEDUP_PARALLELISM = 10`）
3. **合并**：所有批次结果合并后统一处理

### 修改内容

**文件**: `graphiti_core/utils/maintenance/node_operations.py`

```python
# Configuration for batched LLM deduplication
DEDUP_BATCH_SIZE = 5  # Number of entities per LLM request
DEDUP_PARALLELISM = 10  # Maximum concurrent LLM requests


async def _resolve_with_llm(...) -> None:
    """
    EasyOps customization: Process entities in batches to avoid LLM output truncation.
    Each batch contains up to DEDUP_BATCH_SIZE entities, processed with DEDUP_PARALLELISM concurrency.
    """
    # ... existing setup code ...

    # Split entities into batches
    batches: list[list[tuple[int, EntityNode]]] = []
    current_batch: list[tuple[int, EntityNode]] = []
    for global_idx, node in enumerate(llm_extracted_nodes):
        current_batch.append((global_idx, node))
        if len(current_batch) >= DEDUP_BATCH_SIZE:
            batches.append(current_batch)
            current_batch = []
    if current_batch:
        batches.append(current_batch)

    async def process_batch(batch, batch_idx) -> list[tuple[int, NodeDuplicateWithScores]]:
        """Process a single batch and return results with global indices."""
        # Build batch-specific context with local IDs (0 to batch_size-1)
        # Call LLM
        # Map local IDs back to global indices
        ...

    # Process batches in parallel with semaphore
    batch_results = await semaphore_gather(
        *[process_batch(batch, idx) for idx, batch in enumerate(batches)],
        max_coroutines=DEDUP_PARALLELISM,
    )

    # Merge and process all results
    ...
```

### 效果

| 场景 | 优化前 | 优化后 |
|-----|-------|-------|
| 27 个实体 | 1 次 LLM 调用，输出 55000+ 字符 | 6 次并行调用，每次 ~8000 字符 |
| 错误率 | JSON 截断导致失败 | 正常完成 |
| 总耗时 | N/A（失败） | 约等于单次调用（并行处理） |

### 配置参数

| 参数 | 默认值 | 说明 |
|-----|-------|------|
| `DEDUP_BATCH_SIZE` | 5 | 每个 LLM 请求处理的实体数量 |
| `DEDUP_PARALLELISM` | 10 | 最大并发 LLM 请求数 |

---

## 17. RediSearch 反引号转义修复

### 问题背景

导入包含代码片段的文档时，如果内容中包含反引号（Markdown 代码标记），在全文搜索时会导致 RediSearch 语法错误：

```
RediSearch: Syntax error at offset 115 near URL
```

错误的查询示例：
```
(user | info | API | URL | ` | ` | embed | dynamic | values)
```

### 修改内容

**文件**: `graphiti_core/helpers.py`

在 `lucene_sanitize()` 函数的 `escape_map` 中添加反引号转义：

```python
escape_map = str.maketrans(
    {
        # ... 其他字符
        '`': r'\`',  # EasyOps: escape backtick for RediSearch
        # ...
    }
)
```

### 修复效果

| 场景 | 修复前 | 修复后 |
|-----|-------|-------|
| 包含 Markdown 代码的文档 | RediSearch 语法错误 | 正常搜索 |

---

## 24. 实体去重流程优化 - 单轮 LLM 去重（2025-12-18）

### 问题背景

原有的 `add_episode_bulk` 批量导入流程中，实体去重经过**三轮**处理：

1. **第一轮（NodeResolutionsWithScores）**：在 `resolve_extracted_nodes` 中，对每个 episode 的节点与数据库候选进行 LLM 去重，使用 `nodes_with_scores` 提示词
2. **第二轮（确定性匹配）**：在 `dedupe_nodes_bulk` 中，对批次内节点进行精确字符串和 MinHash 匹配
3. **第三轮（NodeResolutions）**：在 `semantic_dedupe_nodes_bulk` 中，attributes 提取后对同类型节点进行语义去重

**问题分析**：

1. **第一轮 LLM 调用时 summary 为空**：此时还没有调用 `extract_attributes_from_nodes`，节点没有 summary 和 attributes，LLM 只能根据 name 判断
2. **第三轮才有完整信息**：此时节点已经有 summary 和 attributes，LLM 可以做出更准确的判断
3. **重复工作**：第一轮和第三轮都调用 LLM 做语义去重，存在重复

**性能分析**（trace c08ad7a370c37014d082d6a9a5d81bf9）：

- 第一轮 LLM 调用：12 次，耗时 280 秒（06:02:08 - 06:06:52）
- 第三轮 LLM 调用：16 次，耗时 40 秒（06:11:28 - 06:12:09）

**第一轮 `NodeResolutionsWithScores` 输出量大**：

每个 LLM 调用需要输出所有候选的相似度评分：
- 5 个节点 × 36 个候选 = 180 组评分数据
- 每组包含 `candidate_idx`, `similarity_score`, `is_same_entity`, `reasoning`

### 解决方案

**去掉第一轮 LLM 去重，只保留第三轮**：

1. `dedupe_nodes_bulk`：只做数据库搜索 + 确定性匹配，收集数据库候选
2. `_resolve_nodes_and_edges_bulk`：去掉 `resolve_extracted_nodes` 调用，只提取 attributes
3. `semantic_dedupe_nodes_bulk`：合并数据库候选和批次内候选，统一做 LLM 去重，**并行执行**

**优化后的流程**：

```
Episode 写入
    │
    ▼
┌─────────────────────────────────────┐
│ 1. extract_nodes_and_edges_bulk     │
│    - LLM 抽取实体和关系              │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 2. dedupe_nodes_bulk (优化后)        │
│    - 数据库搜索（已按 entity type 过滤）│
│    - 确定性匹配（精确字符串 + MinHash）│
│    - 返回 db_candidates_by_type      │
│    - **不调用 LLM**                  │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 3. _resolve_nodes_and_edges_bulk    │
│    - extract_attributes_from_nodes  │
│    - 获取 summary 和 attributes     │
│    - **不调用 resolve_extracted_nodes** │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 4. semantic_dedupe_nodes_bulk (优化后)│
│    - 合并候选：同批次 + 数据库同类型  │
│    - 使用 NodeResolutions（轻量输出）│
│    - **并行处理所有实体**            │
└─────────────────────────────────────┘
    │
    ▼
保存到数据库
```

### 关键优化点

1. **去掉 NodeResolutionsWithScores**：不再需要输出所有候选的评分，改用 NodeResolutions 只返回 duplicate_idx
2. **数据库候选复用**：`dedupe_nodes_bulk` 收集的候选传递给 `semantic_dedupe_nodes_bulk` 使用
3. **串行改并行**：`semantic_dedupe_nodes_bulk` 原来是串行处理每个实体，现在改为并行处理所有实体

### 性能对比

| 指标 | 优化前 | 优化后 |
|-----|-------|-------|
| LLM 去重调用次数 | 第一轮 12 次 + 第三轮 16 次 = 28 次 | 只有一轮，取决于实体数量 |
| LLM 输出格式 | NodeResolutionsWithScores（每候选都打分） | NodeResolutions（只返回 duplicate_idx） |
| 执行方式 | 第三轮串行 | 第三轮并行 |
| 数据库搜索 | 两次（第一轮 + 第三轮） | 一次（第二步收集，第四步复用） |

### 修改文件清单

| 文件 | 修改内容 |
|-----|---------|
| `graphiti_core/utils/bulk_utils.py` | `dedupe_nodes_bulk()` 返回 `db_candidates_by_type`，去掉 `resolve_extracted_nodes` 调用 |
| `graphiti_core/utils/bulk_utils.py` | `semantic_dedupe_nodes_bulk()` 接收 `db_candidates_by_type`，合并候选，并行处理 |
| `graphiti_core/graphiti.py` | `_extract_and_dedupe_nodes_bulk()` 返回 `db_candidates_by_type` |
| `graphiti_core/graphiti.py` | `_resolve_nodes_and_edges_bulk()` 去掉 `resolve_extracted_nodes` 调用 |
| `graphiti_core/graphiti.py` | `add_episode_bulk()` 传递 `db_candidates_by_type` 给 `semantic_dedupe_nodes_bulk()` |

### 接口变更

**`dedupe_nodes_bulk()` 返回值变更**：

```python
# 优化前
async def dedupe_nodes_bulk(...) -> tuple[dict[str, list[EntityNode]], dict[str, str]]:

# 优化后
async def dedupe_nodes_bulk(...) -> tuple[dict[str, list[EntityNode]], dict[str, str], dict[str, list[EntityNode]]]:
    # 第三个返回值: db_candidates_by_type
```

**`semantic_dedupe_nodes_bulk()` 参数变更**：

```python
# 优化前
async def semantic_dedupe_nodes_bulk(
    clients: GraphitiClients,
    nodes: list[EntityNode],
    entity_types: dict[str, type[BaseModel]] | None = None,
) -> list[EntityNode]:

# 优化后
async def semantic_dedupe_nodes_bulk(
    clients: GraphitiClients,
    nodes: list[EntityNode],
    entity_types: dict[str, type[BaseModel]] | None = None,
    db_candidates_by_type: dict[str, list[EntityNode]] | None = None,  # 新增
) -> tuple[list[EntityNode], dict[str, str]]:  # 返回 (nodes, uuid_map)
```

### 回滚说明（2025-12-18）

**此优化已回滚**，恢复了 `resolve_extracted_nodes` 调用。

**回滚原因**：为实现延迟去重策略（Delayed Deduplication），需要恢复 Graphiti 原生的 LLM 去重流程。

**延迟去重策略**：
1. **快速写入**：使用 `resolve_extracted_nodes` 进行实时 LLM 去重（确定性匹配 + LLM 语义判断）
2. **后台维护**：后续实现批量合并脚本，通过 Cypher 语句合并漏网的重复实体

**回滚内容**：
1. `_resolve_nodes_and_edges_bulk`：恢复 `resolve_extracted_nodes` 调用
2. `add_episode_bulk`：注释掉 `semantic_dedupe_nodes_bulk` 调用

**当前去重流程**：
```
Episode 写入
    │
    ▼
┌─────────────────────────────────────┐
│ 1. extract_nodes_and_edges_bulk     │
│    - LLM 抽取实体和关系              │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 2. dedupe_nodes_bulk                │
│    - 数据库搜索                      │
│    - 确定性匹配（精确字符串 + MinHash）│
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 3. _resolve_nodes_and_edges_bulk    │
│    - **resolve_extracted_nodes**    │
│      (确定性 + LLM 语义去重)         │
│    - extract_attributes_from_nodes  │
└─────────────────────────────────────┘
    │
    ▼
保存到数据库
```

---

## 25. 反思环节负向反思功能（Negative Reflexion）

### 问题背景

原有的 `reflexion` 机制只做"正向反思"——检查是否有遗漏的实体或关系，但没有"负向反思"功能来识别：
1. 不适合成为实体的对象（应该删除）
2. 类型分类错误的实体（应该重分类）
3. 不正确的关系（应该删除或修正）

这导致低质量的实体和错误的关系会被写入知识图谱，影响数据质量。

### 解决方案

在 **同一个** reflexion LLM 调用中，同时执行正向和负向反思，不增加 LLM 调用次数。

#### 25.1 扩展实体反思模型

**文件**: `graphiti_core/prompts/extract_nodes.py`

```python
class MissedEntities(BaseModel):
    # 正向反思：遗漏的实体
    missed_entities: list[str] = Field(
        default_factory=list,
        description="Names of entities that weren't extracted but should be"
    )
    # 负向反思：不该抽取的实体
    entities_to_remove: list[str] = Field(
        default_factory=list,
        description='Names of extracted entities that should NOT be in the knowledge graph '
        '(e.g., too generic, transient concepts, document artifacts)',
    )
    # 负向反思：需要重分类的实体
    entities_to_reclassify: list[EntityReclassification] = Field(
        default_factory=list,
        description='Entities that were misclassified and should be assigned a different type',
    )
```

#### 25.2 扩展边反思模型

**文件**: `graphiti_core/prompts/extract_edges.py`

```python
class FactCorrection(BaseModel):
    """Correction for a misextracted fact."""
    original_fact: str = Field(..., description='The original fact text that needs correction')
    issue: str = Field(
        ...,
        description='Type of issue: "wrong_relation_type", "wrong_direction", "nonexistent_relationship"'
    )
    corrected_relation_type: str | None = Field(
        None,
        description='The correct relation_type if issue is "wrong_relation_type"'
    )
    reason: str = Field(..., description='Brief explanation of why this fact is incorrect')


class MissingFacts(BaseModel):
    # 正向反思：遗漏的事实
    missing_facts: list[str] = Field(
        default_factory=list,
        description="Facts that weren't extracted but should be"
    )
    # 负向反思：不该抽取的事实
    facts_to_remove: list[str] = Field(
        default_factory=list,
        description='Facts that should be REMOVED because they are incorrect '
        '(e.g., relationship does not exist, hallucinated, or misinterpreted)',
    )
    # 负向反思：需要修正的事实
    facts_to_correct: list[FactCorrection] = Field(
        default_factory=list,
        description='Facts with wrong relation_type or direction that need correction',
    )
```

#### 25.3 更新实体 reflexion 提示词

**文件**: `graphiti_core/prompts/extract_nodes.py`

```python
def reflexion(context: dict[str, Any]) -> list[Message]:
    sys_prompt = """You are an AI assistant that reviews entity extraction quality. You perform both:
1. **Positive reflexion**: Identify entities that SHOULD have been extracted but weren't
2. **Negative reflexion**: Identify entities that should NOT have been extracted or were misclassified

Review each extracted entity against the Knowledge Graph Builder's Principles:

1. **Permanence Principle**: Does it have lasting value beyond this document?
2. **Connectivity Principle**: Can it meaningfully connect to other entities?
3. **Independence Principle**: Is the name self-explanatory without the source text?
4. **Domain Value Principle**: Does it represent real domain knowledge, not document artifacts?

An entity should be REMOVED if it fails ANY of these principles AND cannot be reclassified to a valid type.

**CRITICAL**: Do NOT trust pre-assigned types blindly. You MUST re-validate the entity against the type's definition in VALID ENTITY TYPES."""

    # ... (参考 filter_entities 的 entity_types 构建逻辑)
```

#### 25.4 更新边 reflexion 提示词

**文件**: `graphiti_core/prompts/extract_edges.py`

```python
def reflexion(context: dict[str, Any]) -> list[Message]:
    sys_prompt = """You are an AI assistant that reviews fact/relationship extraction quality. You perform both:
1. **Positive reflexion**: Identify facts that SHOULD have been extracted but weren't
2. **Negative reflexion**: Identify facts that are INCORRECT and should be removed or corrected

A fact should be REMOVED if:
- The relationship does not actually exist between the entities
- The fact was hallucinated or misinterpreted from the text
- The fact is redundant with another extracted fact

A fact should be CORRECTED if:
- The relation_type is wrong (e.g., WORKS_AT instead of MANAGES)
- The direction is reversed (source and target swapped)"""
```

#### 25.5 修改 node_operations.py 处理负向反思

**文件**: `graphiti_core/utils/maintenance/node_operations.py`

```python
async def extract_nodes_reflexion(
    llm_client: LLMClient,
    episode: EpisodicNode,
    previous_episodes: list[EpisodicNode],
    node_names: list[str],
    group_id: str | None = None,
    entity_types_context: list[dict] | None = None,  # 新增：传入类型定义
) -> MissedEntities:  # 返回完整的反思结果
    """Perform reflexion on extracted entities - both positive and negative."""
    context = {
        'episode_content': episode.content,
        'previous_episodes': [ep.content for ep in previous_episodes],
        'extracted_entities': node_names,
        'entity_types': entity_types_context or [],
    }
    # ...
```

调用处理：

```python
reflexion_result = await extract_nodes_reflexion(...)

# 处理负向反思：移除不该抽取的实体
if reflexion_result.entities_to_remove:
    entities_to_remove_set = set(reflexion_result.entities_to_remove)
    extracted_entities = [e for e in extracted_entities if e.name not in entities_to_remove_set]
    logger.info(f'Negative reflexion removed entities: {reflexion_result.entities_to_remove}')

# 处理重分类
if reflexion_result.entities_to_reclassify:
    reclassify_map = {r.name: r.new_type for r in reflexion_result.entities_to_reclassify}
    # 更新实体类型...

# 处理正向反思：下一轮抽取遗漏的实体
entities_missed = len(reflexion_result.missed_entities) != 0
```

#### 25.6 修改 edge_operations.py 处理负向反思

**文件**: `graphiti_core/utils/maintenance/edge_operations.py`

```python
reflexion_result = MissingFacts(**reflexion_response)

# 处理负向反思：移除不该抽取的事实
if reflexion_result.facts_to_remove:
    facts_to_remove_set = set(reflexion_result.facts_to_remove)
    edges_data = [e for e in edges_data if e.fact not in facts_to_remove_set]
    logger.info(f'Negative reflexion removed facts: {reflexion_result.facts_to_remove}')

# 处理事实修正：更新 relation_type
if reflexion_result.facts_to_correct:
    correction_map = {c.original_fact: c for c in reflexion_result.facts_to_correct}
    for edge_data in edges_data:
        if edge_data.fact in correction_map:
            correction = correction_map[edge_data.fact]
            if correction.issue == 'wrong_relation_type' and correction.corrected_relation_type:
                edge_data.relation_type = correction.corrected_relation_type
            elif correction.issue == 'nonexistent_relationship':
                # 标记为删除
                ...

# 处理正向反思
missing_facts = reflexion_result.missing_facts
```

### 关键设计原则

1. **不增加 LLM 调用次数**：在同一个 reflexion 调用中完成正向和负向反思
2. **复用经过验证的提示词**：负向反思参考 `filter_entities` 的提示词设计
3. **渐进式处理**：先移除不该抽取的，再重分类错误的，最后提取遗漏的

### 日志输出

- `Negative reflexion removed 3 entities: ['it', 'the system', 'this thing']`
- `Reclassified entity "xxx" to type "Feature"`
- `Negative reflexion removed 2 facts: ['incorrect fact 1', 'incorrect fact 2']`
- `Corrected relation_type for fact "xxx": WORKS_AT -> MANAGES`

### 修改文件清单

| 文件 | 修改内容 |
|-----|---------|
| `graphiti_core/prompts/extract_nodes.py` | `MissedEntities` 添加 `entities_to_remove` 和 `entities_to_reclassify` 字段 |
| `graphiti_core/prompts/extract_nodes.py` | `reflexion()` 提示词支持负向反思 |
| `graphiti_core/prompts/extract_edges.py` | 新增 `FactCorrection` 模型 |
| `graphiti_core/prompts/extract_edges.py` | `MissingFacts` 添加 `facts_to_remove` 和 `facts_to_correct` 字段 |
| `graphiti_core/prompts/extract_edges.py` | `reflexion()` 提示词支持负向反思 |
| `graphiti_core/utils/maintenance/node_operations.py` | `extract_nodes_reflexion()` 接收 `entity_types_context`，返回完整 `MissedEntities` |
| `graphiti_core/utils/maintenance/node_operations.py` | `extract_nodes()` 处理负向反思结果 |
| `graphiti_core/utils/maintenance/edge_operations.py` | `extract_edges()` 处理负向反思结果 |

---

## 26. 批量 Summary 提取优化（Batch Summary Extraction）

### 问题背景

分析 trace `f569159d439e463f446ca6212b00efc8` 发现，`extract_attributes_from_nodes` 是性能瓶颈：

| 阶段 | LLM 调用次数 | 平均耗时 | 总耗时 |
|-----|------------|---------|-------|
| EntitySummary (extract_summary) | 322 次 | ~53.5s | **287 分钟** |
| NodeResolutionsWithScores (dedup) | 49 次 | ~7.8s | 6.4 分钟 |

**根因**：`extract_attributes_from_nodes` 对每个节点单独调用 `_extract_entity_summary`，产生 N 次 LLM 调用。

### 解决方案

将单独的 summary 提取改为批量提取，每批 10 个实体合并为 1 次 LLM 调用。

#### 26.1 新增批量提取模型

**文件**: `graphiti_core/prompts/extract_nodes.py`

```python
# EasyOps: Batch summary extraction models
class EntitySummaryItem(BaseModel):
    """Summary for a single entity in batch extraction."""
    entity_id: int = Field(..., description='The ID of the entity from the input list')
    summary: str = Field(
        ...,
        description=f'Summary containing the important information about the entity. Under {MAX_SUMMARY_CHARS} characters.',
    )


class EntitySummaries(BaseModel):
    """Batch of entity summaries."""
    summaries: list[EntitySummaryItem] = Field(
        ..., description='List of summaries for each entity'
    )
```

#### 26.2 新增批量提取提示词

**文件**: `graphiti_core/prompts/extract_nodes.py`

```python
def extract_summaries_bulk(context: dict[str, Any]) -> list[Message]:
    """Extract summaries for multiple entities in a single LLM call.

    Context should contain:
    - entities: list of dicts with 'id', 'name', 'summary' (existing), 'entity_types', 'attributes'
    - episode_content: current episode content
    - previous_episodes: list of previous episode contents
    """
    return [
        Message(
            role='system',
            content='You are a helpful assistant that extracts entity summaries from the provided text. '
            'You will process multiple entities at once and return a summary for each.',
        ),
        Message(
            role='user',
            content=f"""
Given the MESSAGES and the list of ENTITIES, update the summary for EACH entity.
...
For each entity in the list above, provide a summary. Return a JSON object with a "summaries" array.
Each item in the array must have:
- "entity_id": the ID from the input entity
- "summary": the updated summary for that entity

Process ALL entities in the list. Do not skip any entity.
""",
        ),
    ]
```

#### 26.3 实现批量提取函数

**文件**: `graphiti_core/utils/maintenance/node_operations.py`

新增配置常量：

```python
# EasyOps: Batch size for summary extraction
# Process multiple entities in a single LLM call to reduce API calls
SUMMARY_BATCH_SIZE = 10
```

新增批量提取函数：

```python
async def _extract_entity_summaries_bulk(
    llm_client: LLMClient,
    nodes: list[EntityNode],
    episode: EpisodicNode | None,
    previous_episodes: list[EpisodicNode] | None,
    should_summarize_node: NodeSummaryFilter | None,
) -> None:
    """Extract summaries for multiple entities in a single LLM call.

    This reduces LLM API calls from N to N/SUMMARY_BATCH_SIZE.
    Each batch is processed in parallel with other batches.
    """
    # ... implementation with fallback to individual extraction on error
```

#### 26.4 修改 extract_attributes_from_nodes

**文件**: `graphiti_core/utils/maintenance/node_operations.py`

```python
async def extract_attributes_from_nodes(...) -> list[EntityNode]:
    """Extract attributes and summaries from nodes.

    EasyOps optimization: Uses batch summary extraction to reduce LLM calls.
    - Attributes: extracted in parallel (each node needs its own entity_type)
    - Summaries: extracted in batches of SUMMARY_BATCH_SIZE
    """
    # Step 1: Extract attributes in parallel
    await semaphore_gather(
        *[
            _extract_attributes_and_update(llm_client, node, episode, previous_episodes, entity_type)
            for node in nodes
        ]
    )

    # Step 2: Extract summaries in batches
    batches = [nodes[i:i+SUMMARY_BATCH_SIZE] for i in range(0, len(nodes), SUMMARY_BATCH_SIZE)]
    await semaphore_gather(
        *[
            _extract_entity_summaries_bulk(llm_client, batch, episode, previous_episodes, should_summarize_node)
            for batch in batches
        ]
    )

    await create_entity_node_embeddings(embedder, nodes)
    return nodes
```

### 性能对比

| 指标 | 优化前 | 优化后 |
|-----|-------|-------|
| Summary LLM 调用次数 | N 次（每实体 1 次） | ceil(N/10) 次 |
| 322 个实体的调用次数 | 322 次 | 33 次 |
| 预期耗时降低 | - | ~90% |

### 容错机制

批量提取失败时自动降级为单独提取：

```python
try:
    response = await llm_client.generate_response(...)
    # parse and update summaries
except Exception as e:
    logger.error(f'[bulk_summary] Failed to extract summaries in batch: {e}')
    # Fallback to individual extraction
    for node in nodes_to_process:
        await _extract_entity_summary(llm_client, node, episode, previous_episodes, should_summarize_node)
```

### 日志输出

- `[bulk_summary] Processing 322 nodes in 33 batches (batch_size=10)`
- `[bulk_summary] Extracted summaries for 10/10 entities`
- `[bulk_summary] Falling back to individual extraction for 10 entities` (失败时)

### 修改文件清单

| 文件 | 修改内容 |
|-----|---------|
| `graphiti_core/prompts/extract_nodes.py` | 新增 `EntitySummaryItem` 和 `EntitySummaries` 模型 |
| `graphiti_core/prompts/extract_nodes.py` | 新增 `extract_summaries_bulk()` 提示词函数 |
| `graphiti_core/prompts/extract_nodes.py` | 更新 `Prompt` Protocol 和 `Versions` TypedDict |
| `graphiti_core/utils/maintenance/node_operations.py` | 新增 `SUMMARY_BATCH_SIZE` 常量 |
| `graphiti_core/utils/maintenance/node_operations.py` | 新增 `_extract_entity_summaries_bulk()` 函数 |
| `graphiti_core/utils/maintenance/node_operations.py` | 新增 `_extract_attributes_and_update()` 辅助函数 |
| `graphiti_core/utils/maintenance/node_operations.py` | 重构 `extract_attributes_from_nodes()` 使用批量提取 |

---

## 两轮 Map-Reduce 语义去重（Semantic Dedup with Map-Reduce）

### 问题背景

之前的去重流程存在问题：

1. `resolve_extracted_nodes` 在 `_resolve_nodes_and_edges_bulk` 中被调用
2. 此时节点还没有 summary，LLM 只能看到 name，决策质量差
3. 是串行执行的（一个 episode 一个 episode 处理）

### 解决方案

1. **移除** `_resolve_nodes_and_edges_bulk` 中的 `resolve_extracted_nodes` 调用
2. **启用** `semantic_dedupe_nodes_bulk`，在 `extract_attributes_from_nodes` 之后执行
3. **改为两轮 Map-Reduce**：分批并行 + 跨批去重

### 修改后的流程

```
add_episode_bulk
  ↓
_extract_and_dedupe_nodes_bulk
  └─ dedupe_nodes_bulk (确定性去重)
  ↓
_resolve_nodes_and_edges_bulk
  ├─ 【已删除】resolve_extracted_nodes
  └─ extract_attributes_from_nodes (提取 summary)
  ↓
semantic_dedupe_nodes_bulk (两轮 Map-Reduce，有 summary)
  ↓
add_nodes_and_edges_bulk (保存)
```

### Map-Reduce 设计

#### 第一轮：Map（分批并行去重）

```
输入: 100 个节点（按类型分组）
     ↓
分批: [batch_1: 10个], [batch_2: 10个], ..., [batch_10: 10个]
     ↓ 并行执行
每批任务:
  - 批内节点相互比较 → LLM 判断重复
  - 批内节点 vs DB 候选 → LLM 判断重复
     ↓
输出: 每批的 (去重后节点, duplicate_pairs)
```

#### 第二轮：Reduce（跨批去重）

```
输入: 10 个批次的代表节点
     ↓
跨批去重: LLM 判断不同批次的节点是否重复
     ↓
输出: 最终的 uuid_map（使用并查集合并）
```

### 新增函数

```python
async def _map_batch_dedup(
    llm_client: LLMClient,
    batch_nodes: list[EntityNode],
    db_candidates: list[EntityNode],
    entity_types: dict[str, type[BaseModel]] | None,
) -> tuple[list[EntityNode], list[tuple[str, str]]]:
    """Map phase: 单批次去重（批内 + DB 候选）"""

async def _reduce_cross_batch_dedup(
    llm_client: LLMClient,
    representative_nodes: list[EntityNode],
    entity_types: dict[str, type[BaseModel]] | None,
) -> tuple[list[EntityNode], list[tuple[str, str]]]:
    """Reduce phase: 跨批次去重"""

def _build_uuid_map_from_pairs(
    duplicate_pairs: list[tuple[str, str]]
) -> dict[str, str]:
    """使用并查集处理链式映射和冲突"""
```

### 冲突处理

使用并查集处理：
- `A->B, B->C` 折叠为 `A->C, B->C`
- `A->B, A->C` 保留第一个（先到先得）

### 配置参数

- `batch_size`: 每批节点数，默认 10

### 修改文件清单

| 文件 | 修改内容 |
|-----|---------|
| `graphiti_core/graphiti.py` | 删除 `resolve_extracted_nodes` 调用 |
| `graphiti_core/graphiti.py` | 启用 `semantic_dedupe_nodes_bulk` 调用 |
| `graphiti_core/utils/bulk_utils.py` | 重写 `semantic_dedupe_nodes_bulk` 为 Map-Reduce |
| `graphiti_core/utils/bulk_utils.py` | 新增 `_map_batch_dedup()` 函数 |
| `graphiti_core/utils/bulk_utils.py` | 新增 `_reduce_cross_batch_dedup()` 函数 |
| `graphiti_core/utils/bulk_utils.py` | 新增 `_build_uuid_map_from_pairs()` 函数 |

---

## 实体去重 LLM 调用优化（Dedup LLM Call Optimization）

### 问题背景

原有的 Map-Reduce 去重存在严重的性能问题：

1. **全局 DB 候选收集**：在去重前收集所有节点的 DB 候选（每节点搜 20 个），导致大量 DB 候选被收集但实际只用一部分
2. **O(n²) 批内去重**：每个节点单独调用 LLM 对比后续节点，N 个节点需要 N 次 LLM 调用
3. **批内去重 + DB 匹配分离**：分两步处理，增加了 LLM 调用次数

**实际案例**：91 个 Feature 实体产生了 186 次 LLM 调用（理论上应该远少于此）。

### 解决方案

#### 1. 每批独立搜索 DB 候选

**修改前**：
```python
# 全局收集所有节点的 DB 候选
db_search_results = await semaphore_gather(
    *[_collect_candidate_nodes(clients, nodes, None) for nodes in extracted_nodes]
)
# 传给每个 batch
for batch in batches:
    await _map_batch_dedup(llm_client, batch, filtered_db_candidates, ...)
```

**修改后**：
```python
async def _map_batch_dedup(clients, batch_nodes, entity_types):
    # 每个 batch 独立搜索自己的 DB 候选
    db_candidates = await _search_batch_db_candidates(clients, batch_nodes)
    # 一次 LLM 调用
    ...
```

**新增配置**：
```python
# 每节点搜索 10 个候选（原来是 20）
DEDUP_SEARCH_LIMIT = 10
DEDUP_SEARCH_CONFIG = SearchConfig(
    limit=DEDUP_SEARCH_LIMIT,
    node_config=NodeSearchConfig(
        search_methods=[NodeSearchMethod.bm25, NodeSearchMethod.cosine_similarity],
        reranker=NodeReranker.rrf,
    )
)
```

#### 2. 合并批内去重 + DB 匹配为单次 LLM 调用

**修改前**：
```python
async def _map_batch_dedup(llm_client, batch_nodes, db_candidates, ...):
    # 步骤1: 批内去重 - 每个节点单独调 LLM（O(n²)）
    for i, node in enumerate(batch_nodes):
        candidates = [n for n in batch_nodes[i + 1:] if ...]
        if candidates:
            llm_pairs = await _resolve_batch_with_llm(llm_client, [node], candidates, ...)

    # 步骤2: DB 匹配 - 又一次 LLM 调用
    if remaining_nodes and db_candidates:
        llm_pairs = await _resolve_batch_with_llm(llm_client, remaining_nodes, db_candidates, ...)
```

**修改后**：
```python
async def _map_batch_dedup(clients, batch_nodes, entity_types):
    db_candidates = await _search_batch_db_candidates(clients, batch_nodes)

    # 所有候选 = DB 候选 + batch 其他节点（DB 放前面表示优先）
    all_candidates = db_candidates + batch_nodes

    # 一次 LLM 调用搞定
    llm_pairs = await _resolve_batch_with_llm(
        clients.llm_client, batch_nodes, all_candidates, entity_types
    )
```

#### 3. Reduce 阶段简化

**修改前**：
```python
async def _reduce_cross_batch_dedup(llm_client, representative_nodes, ...):
    # 每个节点单独调 LLM（O(n²)）
    for i, node in enumerate(representative_nodes):
        candidates = [n for n in representative_nodes[i + 1:] if ...]
        if candidates:
            llm_pairs = await _resolve_batch_with_llm(llm_client, [node], candidates, ...)
```

**修改后**：
```python
async def _reduce_cross_batch_dedup(llm_client, representative_nodes, ...):
    # 一次 LLM 调用
    llm_pairs = await _resolve_batch_with_llm(
        llm_client, representative_nodes, representative_nodes, entity_types
    )
```

#### 4. 移除全局 DB 候选收集

`dedupe_nodes_bulk` 不再返回 `db_candidates_by_type`，函数签名从：
```python
async def dedupe_nodes_bulk(...) -> tuple[dict, dict, dict]:
```
改为：
```python
async def dedupe_nodes_bulk(...) -> tuple[dict, dict]:
```

### 优化效果

| 指标 | 修改前 | 修改后 |
|-----|-------|-------|
| Map 阶段 LLM 调用 | N × batch_count 次 | batch_count 次 |
| Reduce 阶段 LLM 调用 | N × (N-1)/2 次 | 1 次 |
| 每节点 DB 候选数 | 20 | 10 |
| 候选收集时机 | 全局预收集 | 按批实时搜索 |

**预期效果**：91 个 Feature 实体从 186 次 LLM 调用减少到约 10 次（10 个 batch × 1 次 + 1 次 Reduce）。

### 修改文件清单

| 文件 | 修改内容 |
|-----|---------|
| `graphiti_core/utils/bulk_utils.py` | 新增 `DEDUP_SEARCH_LIMIT`、`DEDUP_SEARCH_CONFIG` 常量 |
| `graphiti_core/utils/bulk_utils.py` | 新增 `_search_batch_db_candidates()` 函数 |
| `graphiti_core/utils/bulk_utils.py` | 重写 `_map_batch_dedup()` - 每批独立搜候选 + 单次 LLM |
| `graphiti_core/utils/bulk_utils.py` | 重写 `_reduce_cross_batch_dedup()` - 单次 LLM |
| `graphiti_core/utils/bulk_utils.py` | 重写 `semantic_dedupe_nodes_bulk()` - 移除 db_candidates_by_type 参数 |
| `graphiti_core/utils/bulk_utils.py` | 重写 `dedupe_nodes_bulk()` - 移除 DB 候选收集和返回 |
| `graphiti_core/graphiti.py` | 更新 `_extract_and_dedupe_nodes_bulk()` - 移除 db_candidates_by_type |
| `graphiti_core/graphiti.py` | 更新 `add_episodes_bulk()` - 移除 db_candidates_by_type 传递 |

---

## 离线清理脚本（Offline Cleanup Script）

### 问题背景

两轮 Map-Reduce 语义去重只能处理：
1. 新节点 vs 新节点
2. 新节点 vs 数据库候选

但无法处理：数据库中已存在的重复节点（历史数据）。

如果数据库中已经存在多个重复的 ProductModule 节点（如 `EasyITSM`、`IT服务中心`、`ITSC服务中心`），这些节点不会被自动合并。

### 解决方案

提供离线清理脚本 `scripts/cleanup_duplicate_nodes.py`，使用 LLM 识别并合并数据库中的重复实体。

### 使用方法

```bash
# 干运行（不做实际修改，只显示会做什么）
python scripts/cleanup_duplicate_nodes.py --group-id easyops_support --dry-run

# 清理特定实体类型
python scripts/cleanup_duplicate_nodes.py --group-id easyops_support --entity-type ProductModule

# 清理所有实体类型
python scripts/cleanup_duplicate_nodes.py --group-id easyops_support

# 指定数据库连接
python scripts/cleanup_duplicate_nodes.py \
    --group-id easyops_support \
    --falkordb-host localhost \
    --falkordb-port 6379 \
    --falkordb-database elevo_memory
```

### 工作流程

1. **查询节点**：从数据库查询指定 group_id 的所有实体节点，按类型分组
2. **识别重复**：对每种类型的节点，使用 LLM 识别语义重复的节点对
3. **合并节点**：
   - 将源节点的所有边重定向到规范节点
   - 合并源节点的 summary 和 attributes 到规范节点
   - 删除源节点

### 环境变量

| 变量 | 说明 |
|-----|------|
| `OPENAI_API_KEY` | LLM API Key |
| `OPENAI_MODEL` | LLM 模型名称 |
| `OPENAI_BASE_URL` | LLM API 端点（可选） |
| `FALKORDB_HOST` | FalkorDB 主机 |
| `FALKORDB_PORT` | FalkorDB 端口 |
| `FALKORDB_DATABASE` | 数据库名称 |

### 注意事项

1. **先干运行**：建议先用 `--dry-run` 查看会合并哪些节点
2. **备份数据**：执行前建议备份数据库
3. **LLM 成本**：每对节点比较都需要调用 LLM，大量节点会产生较高成本
4. **执行时间**：O(n²) 复杂度，节点数多时耗时较长

### 修改文件清单

| 文件 | 修改内容 |
|-----|---------|
| `scripts/cleanup_duplicate_nodes.py` | 新增离线清理脚本 |

---

## 27. 批量去重别名匹配修复（Alias-Based Deduplication）

### 问题背景

批量导入时，Schema 中定义的别名实体（如 `EasyITSM 或 IT服务中心`）未被正确去重，导致同一实体创建多个节点。

**具体案例**：

ProductModule 类型定义包含别名模式：
```
(5)EasyITSM 或 ITSM 或 IT服务中心
```

预期：`EasyITSM`、`ITSM`、`IT服务中心` 应被识别为同一实体的不同名称。

实际：批量导入时这三个名称分别创建了独立节点。

### 根因分析

1. **`_resolve_batch_with_llm()` 未传递 `entity_type_definitions`**

   每个节点的 `entity_type_description` 包含别名信息，但 context 构建时没有传递 `entity_type_definitions`：

   ```python
   # 修改前 (bulk_utils.py:145-150)
   context = {
       'extracted_nodes': extracted_nodes_context,
       'existing_nodes': existing_nodes_context,
       'episode_content': '',
       'previous_episodes': [],
       # 缺失: 'entity_type_definitions' 未传递！
   }
   ```

2. **提示词未指导 LLM 使用别名信息**

   `dedupe_nodes.py` 的 `nodes()` 函数：
   - 期望 `entity_type_definitions` 但始终为空
   - 没有说明如何使用 `entity_type_description` 中的别名模式
   - 结果：LLM 只按字面名称匹配，不识别别名关系

### 解决方案

#### 27.1 传递 entity_type_definitions

**文件**: `graphiti_core/utils/bulk_utils.py`

在 `_resolve_batch_with_llm()` 中构建并传递 `entity_type_definitions`：

```python
entity_types_dict = entity_types or {}

# Build entity_type_definitions for the prompt (EasyOps: enables alias-based deduplication)
# This tells LLM about type definitions including alias patterns like "EasyITSM 或 IT服务中心"
entity_type_definitions: dict[str, str] = {}
for node in nodes:
    for label in node.labels:
        if label != 'Entity' and label not in entity_type_definitions:
            type_model = entity_types_dict.get(label)
            if type_model and type_model.__doc__:
                entity_type_definitions[label] = type_model.__doc__

# ... later in context ...
context = {
    'extracted_nodes': extracted_nodes_context,
    'existing_nodes': existing_nodes_context,
    'episode_content': '',
    'previous_episodes': [],
    'entity_type_definitions': entity_type_definitions,  # EasyOps: enables alias detection
}
```

#### 27.2 添加别名匹配指导

**文件**: `graphiti_core/prompts/dedupe_nodes.py`

在 `nodes()` 函数中添加别名匹配的明确指导：

```python
def nodes(context: dict[str, Any]) -> list[Message]:
    type_defs = context.get('entity_type_definitions', {})
    type_defs_section = ''
    alias_instruction = ''
    if type_defs:
        type_defs_section = f"""
        <ENTITY TYPE DEFINITIONS>
        {to_prompt_json(type_defs)}
        </ENTITY TYPE DEFINITIONS>
        """
        # EasyOps: Add explicit alias matching instruction when type definitions are available
        alias_instruction = """
        **IMPORTANT - ALIAS MATCHING**:
        The ENTITY TYPE DEFINITIONS above may contain alias patterns using "或" (Chinese "or").
        For example: "(5)EasyITSM 或 ITSM 或 IT服务中心" means EasyITSM, ITSM, and IT服务中心 are ALL aliases for the SAME entity.
        When you see entities with names matching these aliases, they MUST be marked as duplicates.
        """
```

同时更新实体结构描述，包含 `summary` 和 `entity_type_description` 字段：

```python
Each entity in ENTITIES is represented as a JSON object with the following structure:
{{
    id: integer id of the entity,
    name: "name of the entity",
    summary: "brief description of the entity",
    entity_type: ["Entity", "<optional additional label>", ...],
    entity_type_description: "description of the entity type, may contain alias patterns"
}}
```

### 效果

| 实体名称 | 修复前 | 修复后 |
|---------|-------|-------|
| EasyITSM | 独立节点 | 去重为同一节点 |
| IT服务中心 | 独立节点 | 去重为同一节点 |
| ITSM | 独立节点 | 去重为同一节点 |

LLM 现在能够：
1. 看到 `<ENTITY TYPE DEFINITIONS>` 中的类型定义和别名模式
2. 理解 "或" 表示的别名关系
3. 正确将不同名称但属于同一别名组的实体标记为重复

### 修改文件清单

| 文件 | 修改内容 |
|-----|---------|
| `graphiti_core/utils/bulk_utils.py` | `_resolve_batch_with_llm()` 构建并传递 `entity_type_definitions` |
| `graphiti_core/prompts/dedupe_nodes.py` | `nodes()` 添加 `alias_instruction` 和更完整的实体结构描述 |