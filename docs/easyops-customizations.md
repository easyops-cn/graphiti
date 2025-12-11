# EasyOps Graphiti 自定义修改

本文档记录 EasyOps 对 Graphiti 的自定义修改，便于后续维护和升级时参考。

## 修改概览

| 文件 | 修改类型 | 说明 |
|------|---------|------|
| `graphiti_core/models/edges/edge_db_queries.py` | 新增函数 | 支持动态边类型的 Cypher 查询 |
| `graphiti_core/utils/bulk_utils.py` | 逻辑修改 | 按边类型分组保存 |
| `graphiti_core/utils/bulk_utils.py` | 新增函数 | 批量去重第三轮 LLM 语义去重 |
| `graphiti_core/utils/bulk_utils.py` | 逻辑修改 | `_merge_node_into_canonical()` 记录同义词 |
| `graphiti_core/graphiti.py` | 逻辑修改 | `add_episode()` 记录 duplicate_pairs 同义词 |
| `graphiti_core/graph_queries.py` | 参数修改 | BM25 索引添加 synonyms 字段 |
| `graphiti_core/utils/maintenance/edge_operations.py` | 逻辑修改 | 严格控制边类型，非 Schema 类型降级 |
| `graphiti_core/prompts/extract_nodes.py` | 新增 | Knowledge Graph Builder's Principles + filter_entities |
| `graphiti_core/utils/maintenance/node_operations.py` | 新增 | filter_extracted_nodes 函数 |
| `graphiti_core/llm_client/openai_generic_client.py` | 新增 | small_model 支持 |
| `graphiti_core/utils/maintenance/node_operations.py` | 逻辑修改 | 候选搜索按同实体类型过滤 |
| `graphiti_core/utils/maintenance/edge_operations.py` | 逻辑修改 | 候选搜索按同边类型过滤 |
| `graphiti_core/search/search_config.py` | 参数修改 | DEFAULT_SEARCH_LIMIT 从 10 改为 20 |
| `graphiti_core/prompts/dedupe_edges.py` | 新增字段 | EdgeDuplicate 添加 merged_fact 字段 |
| `graphiti_core/utils/maintenance/edge_operations.py` | 逻辑修改 | 边去重时合并 fact 和属性 |

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
- **synonyms**: 记录被合并实体的名称

---

## 6. 同义词属性与搜索支持（已实现）

### 需求背景

实体去重合并后，被合并实体的名称（同义词）应该被保留，用于搜索时能通过同义词找到实体。

**场景示例**：
- 实体 `IT服务中心` 合并了 `EasyITSM` 后，`synonyms = "EasyITSM"`
- 搜索 "EasyITSM" 时，能命中 `IT服务中心` 这个实体

### 实现方案

#### 6.1 数据存储（空格分隔字符串）

同义词存储为**空格分隔的字符串**（而非数组），以便 BM25 全文索引直接支持。

**文件**: `graphiti_core/utils/bulk_utils.py`

在 `_merge_node_into_canonical()` 中添加同义词和 reasoning 合并：

```python
def _merge_node_into_canonical(source: EntityNode, canonical: EntityNode):
    # 现有合并逻辑（summary, attributes）...

    # EasyOps: Merge reasoning - prefer non-empty, concatenate if both exist
    if source.reasoning and canonical.reasoning:
        if source.reasoning not in canonical.reasoning:
            canonical.reasoning = f"{canonical.reasoning}\n---\n{source.reasoning}"
    elif source.reasoning and not canonical.reasoning:
        canonical.reasoning = source.reasoning

    # EasyOps: Record synonyms (space-separated string for BM25 full-text index)
    if source.name and source.name != canonical.name:
        existing_synonyms = canonical.attributes.get('synonyms', '')
        synonym_list = existing_synonyms.split() if existing_synonyms else []
        if source.name not in synonym_list:
            synonym_list.append(source.name)
            canonical.attributes['synonyms'] = ' '.join(synonym_list)
```

#### 6.2 单条写入时记录同义词

**文件**: `graphiti_core/graphiti.py`

在 `add_episode()` 方法中，`resolve_extracted_nodes()` 返回的 `duplicate_pairs` 包含检测到的重复实体对。
修改代码以记录这些同义词：

```python
(nodes, uuid_map, duplicate_pairs), extracted_edges = await semaphore_gather(
    resolve_task, extract_edges_task
)

# EasyOps: Record synonyms from duplicate_pairs to canonical nodes
for source_node, canonical_node in duplicate_pairs:
    if source_node.name and source_node.name != canonical_node.name:
        existing_synonyms = canonical_node.attributes.get('synonyms', '')
        synonym_list = existing_synonyms.split() if existing_synonyms else []
        if source_node.name not in synonym_list:
            synonym_list.append(source_node.name)
            canonical_node.attributes['synonyms'] = ' '.join(synonym_list)
            logger.info(f'[synonym] Recorded synonym "{source_node.name}" for entity "{canonical_node.name}"')
```

#### 6.3 搜索增强：BM25 全文索引

将 `synonyms` 加入 BM25 全文索引：

**文件**: `graphiti_core/graph_queries.py`

**Neo4j**（第 121-122 行）:
```python
"""CREATE FULLTEXT INDEX node_name_and_summary IF NOT EXISTS
FOR (n:Entity) ON EACH [n.name, n.summary, n.synonyms, n.group_id]"""
```

**FalkorDB**（第 92-98 行）:
```python
f"""CALL db.idx.fulltext.createNodeIndex(
    {{ label: 'Entity', stopwords: {stopwords_str} }},
    'name', 'summary', 'synonyms', 'group_id'
)"""
```

**Kuzu**（第 113 行）:
```python
"CALL CREATE_FTS_INDEX('Entity', 'node_name_and_summary', ['name', 'summary', 'synonyms']);"
```

**注意**：升级后需删除旧索引重建：
```cypher
-- FalkorDB
CALL db.idx.fulltext.drop('Entity')
-- 然后重启服务，索引会自动创建
```

#### 6.4 API 暴露

**文件**: `src/elevo_memory/models/entity.py`

```python
class EntityNode(BaseModel):
    synonyms: list[str] = Field(default_factory=list, description="实体同义词列表")
```

**文件**: `src/elevo_memory/services/query_utils.py`

```python
def graphiti_node_to_entity(node):
    # 提取同义词（空格分隔字符串 -> 列表）
    synonyms_str = node.attributes.get('synonyms', '') if node.attributes else ''
    synonyms = synonyms_str.split() if synonyms_str else []

    return EntityNode(
        synonyms=synonyms,
        ...
    )
```

### 触发范围

| API | 会记录同义词? |
|-----|--------------|
| `add_episode_bulk()` 批量 | ✅ 会（通过 `semantic_dedupe_nodes_bulk`） |
| `add_episode()` 单条 | ✅ 会（通过 `duplicate_pairs`） |
| `triplet` API | ❌ 不会（直接写入，不经过去重） |

### 修改文件清单

| 文件 | 修改内容 |
|-----|---------|
| `graphiti_core/utils/bulk_utils.py` | `_merge_node_into_canonical()` 添加同义词记录 |
| `graphiti_core/graphiti.py` | `add_episode()` 中记录 `duplicate_pairs` 的同义词 |
| `graphiti_core/graph_queries.py` | BM25 索引加入 synonyms 字段（3 个数据库） |
| `src/elevo_memory/models/entity.py` | EntityNode 添加 synonyms 字段 |
| `src/elevo_memory/services/query_utils.py` | 转换时提取 synonyms |

---

## 7. 边去重时合并 Fact（Edge Fact Merging）

### 问题背景

原始 Graphiti 的边去重逻辑存在问题：
1. 当检测到重复边时，直接丢弃新边的 fact，只保留旧边的 fact
2. 不同语言或不同表述的相同关系没有被识别为重复
3. 重复边的属性（valid_at, invalid_at, attributes）没有合并

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

#### 7.3 修改去重逻辑

**文件**: `graphiti_core/utils/maintenance/edge_operations.py`

修改 `resolve_extracted_edge` 函数中的去重逻辑：

```python
for duplicate_fact_id in duplicate_fact_ids:
    existing_edge = related_edges[duplicate_fact_id]
    # 使用 LLM 返回的合并 fact
    merged_fact = response_object.merged_fact
    if merged_fact and merged_fact.strip():
        existing_edge.fact = merged_fact.strip()
        existing_edge.fact_embedding = None  # 清除 embedding，后续重新计算
        logger.info(f'[edge_dedup] Updated fact for edge {existing_edge.uuid} with LLM-merged content')
    # 合并属性
    for key, value in extracted_edge.attributes.items():
        if key not in existing_edge.attributes:
            existing_edge.attributes[key] = value
    # 合并 valid_at：取较早的时间戳
    if extracted_edge.valid_at and existing_edge.valid_at:
        if extracted_edge.valid_at < existing_edge.valid_at:
            existing_edge.valid_at = extracted_edge.valid_at
    # 合并 invalid_at：取较晚的时间戳
    if extracted_edge.invalid_at and existing_edge.invalid_at:
        if extracted_edge.invalid_at > existing_edge.invalid_at:
            existing_edge.invalid_at = extracted_edge.invalid_at
    resolved_edge = existing_edge
    break
```

### 合并策略

| 字段 | 合并策略 |
|-----|---------|
| `fact` | LLM 合并，综合两个 fact 的语义信息 |
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
| `graphiti_core/utils/maintenance/edge_operations.py` | `resolve_extracted_edge` 使用合并 fact |
