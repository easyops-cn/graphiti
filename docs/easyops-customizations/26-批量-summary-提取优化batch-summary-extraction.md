# 26. 批量 Summary 提取优化（Batch Summary Extraction）

## 问题背景

分析 trace `f569159d439e463f446ca6212b00efc8` 发现，`extract_attributes_from_nodes` 是性能瓶颈：

| 阶段 | LLM 调用次数 | 平均耗时 | 总耗时 |
|-----|------------|---------|-------|
| EntitySummary (extract_summary) | 322 次 | ~53.5s | **287 分钟** |
| NodeResolutionsWithScores (dedup) | 49 次 | ~7.8s | 6.4 分钟 |

**根因**：`extract_attributes_from_nodes` 对每个节点单独调用 `_extract_entity_summary`，产生 N 次 LLM 调用。

## 解决方案

将单独的 summary 提取改为批量提取，每批 10 个实体合并为 1 次 LLM 调用。

### 26.1 新增批量提取模型

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

### 26.2 新增批量提取提示词

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

### 26.3 实现批量提取函数

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

### 26.4 修改 extract_attributes_from_nodes

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

## 性能对比

| 指标 | 优化前 | 优化后 |
|-----|-------|-------|
| Summary LLM 调用次数 | N 次（每实体 1 次） | ceil(N/10) 次 |
| 322 个实体的调用次数 | 322 次 | 33 次 |
| 预期耗时降低 | - | ~90% |

## 容错机制

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

## 日志输出

- `[bulk_summary] Processing 322 nodes in 33 batches (batch_size=10)`
- `[bulk_summary] Extracted summaries for 10/10 entities`
- `[bulk_summary] Falling back to individual extraction for 10 entities` (失败时)

## 修改文件清单

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
