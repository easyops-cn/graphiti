# 20. 实体去重分批并行处理（LLM 输出截断修复）

## 问题背景

当一次导入的文档包含大量实体时（如 27 个），实体去重阶段需要将每个新实体与所有候选实体进行相似度评分。LLM 需要为每个新实体的每个候选输出 JSON 格式的评分数据，包含 `candidate_idx`, `similarity_score`, `is_same_entity`, `reasoning` 字段。

问题是：
- 27 个新实体 × 50+ 个候选 = 需要输出几千条评分数据
- 输出 JSON 达到 55000+ 字符
- 超过 LLM 的 `max_output_tokens` 限制，JSON 被截断
- 导致 `JSONDecodeError: Unterminated string` 错误

## 解决方案

将实体分批处理，每批并行调用 LLM：

1. **分批**：每批最多 5 个实体（`DEDUP_BATCH_SIZE = 5`）
2. **并行**：最多 10 个并发请求（`DEDUP_PARALLELISM = 10`）
3. **合并**：所有批次结果合并后统一处理

## 修改内容

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

## 效果

| 场景 | 优化前 | 优化后 |
|-----|-------|-------|
| 27 个实体 | 1 次 LLM 调用，输出 55000+ 字符 | 6 次并行调用，每次 ~8000 字符 |
| 错误率 | JSON 截断导致失败 | 正常完成 |
| 总耗时 | N/A（失败） | 约等于单次调用（并行处理） |

## 配置参数

| 参数 | 默认值 | 说明 |
|-----|-------|------|
| `DEDUP_BATCH_SIZE` | 5 | 每个 LLM 请求处理的实体数量 |
| `DEDUP_PARALLELISM` | 10 | 最大并发 LLM 请求数 |

---
