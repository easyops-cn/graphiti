# EasyOps Graphiti 自定义修改

本文档记录 EasyOps 对 Graphiti 的自定义修改，便于后续维护和升级时参考。

## 修改概览

| 文件 | 修改类型 | 说明 |
|------|---------|------|
| `graphiti_core/models/edges/edge_db_queries.py` | 新增函数 | 支持动态边类型的 Cypher 查询 |
| `graphiti_core/models/edges/edge_db_queries.py` | 逻辑修改 | MERGE 按节点对合并，避免重复边 |
| `graphiti_core/utils/bulk_utils.py` | 逻辑修改 | 按边类型分组保存 |
| `graphiti_core/utils/bulk_utils.py` | 新增函数 | 批量去重第三轮 LLM 语义去重 |
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
