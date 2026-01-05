# 25. 反思环节负向反思功能（Negative Reflexion）

## 问题背景

原有的 `reflexion` 机制只做"正向反思"——检查是否有遗漏的实体或关系，但没有"负向反思"功能来识别：
1. 不适合成为实体的对象（应该删除）
2. 类型分类错误的实体（应该重分类）
3. 不正确的关系（应该删除或修正）

这导致低质量的实体和错误的关系会被写入知识图谱，影响数据质量。

## 解决方案

在 **同一个** reflexion LLM 调用中，同时执行正向和负向反思，不增加 LLM 调用次数。

### 25.1 扩展实体反思模型

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

### 25.2 扩展边反思模型

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

### 25.3 更新实体 reflexion 提示词

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

### 25.4 更新边 reflexion 提示词

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

### 25.5 修改 node_operations.py 处理负向反思

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

### 25.6 修改 edge_operations.py 处理负向反思

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

## 关键设计原则

1. **不增加 LLM 调用次数**：在同一个 reflexion 调用中完成正向和负向反思
2. **复用经过验证的提示词**：负向反思参考 `filter_entities` 的提示词设计
3. **渐进式处理**：先移除不该抽取的，再重分类错误的，最后提取遗漏的

## 日志输出

- `Negative reflexion removed 3 entities: ['it', 'the system', 'this thing']`
- `Reclassified entity "xxx" to type "Feature"`
- `Negative reflexion removed 2 facts: ['incorrect fact 1', 'incorrect fact 2']`
- `Corrected relation_type for fact "xxx": WORKS_AT -> MANAGES`

## 修改文件清单

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
