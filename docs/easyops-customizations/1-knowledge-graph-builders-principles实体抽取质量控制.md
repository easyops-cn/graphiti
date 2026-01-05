# 1. Knowledge Graph Builder's Principles（实体抽取质量控制）

## 问题背景

原始 Graphiti 的实体抽取存在以下问题：
1. 抽取过多噪声实体（代码变量名、占位符、无意义的技术术语）
2. 实体分类准确率较低，大量实体被错误分类为 Feature 或 Entity
3. 原有的 reflexion 机制是"找遗漏"，而不是"过滤噪声"

## 解决方案

### 1.1 添加 Knowledge Graph Builder's Principles

**文件**: `graphiti_core/prompts/extract_nodes.py`

在所有实体抽取提示词中添加四条核心原则：

```python
KNOWLEDGE_GRAPH_PRINCIPLES = """
# KNOWLEDGE GRAPH BUILDER'S PRINCIPLES (Must Follow)

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

### 1.2 添加 filter_entities 提示词

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

### 1.3 添加 filter_extracted_nodes 函数

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

### 1.4 Filter 步骤位置：extract_attributes 之后

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

## 效果

- 减少噪声实体（代码变量、占位符等）的抽取
- **利用 summary 信息做更准确的过滤决策**
- 匹配 Schema 实体类型的实体不会被误删
- 知识图谱质量更高，更有价值

---
