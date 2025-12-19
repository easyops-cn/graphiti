# LLM 调用风险分析：大批量输出截断问题

本文档记录 Graphiti 中可能因大量数据导致 LLM 输出截断的风险点。

## 问题背景

当输入实体/边数量较大时，LLM 需要输出大量 JSON 数据。如果输出超过 `max_output_tokens` 限制，JSON 会被截断，导致 `JSONDecodeError: Unterminated string` 错误。

**实际案例**：27 个实体 × 50+ 候选 = 55000+ 字符的 JSON 输出，超出限制导致截断。

---

## 风险点清单

### 高风险 (已修复)

| 文件 | 函数 | 风险描述 | 状态 |
|-----|------|---------|------|
| `node_operations.py` | `_resolve_with_llm()` | 实体去重，每个实体需要对所有候选打分输出 | ✅ 已修复 - 分批并行处理 |

### 高风险 (待处理)

| 文件 | 函数 | 风险描述 | 建议方案 |
|-----|------|---------|---------|
| `edge_operations.py` | `resolve_extracted_edge()` | 边去重，如果边数量大且候选多，同样会截断 | 参考 node_operations 实现分批处理 |

### 中风险

| 文件 | 函数 | 风险描述 | 备注 |
|-----|------|---------|------|
| `bulk_utils.py` | `semantic_dedupe_nodes_bulk()` | 批量自去重，O(n²) 调用 | 当前是 O(n²) 串行，每次只比 1 个节点，截断风险较低 |
| `node_operations.py` | `extract_nodes()` | 实体抽取，reflexion 迭代 | 输出是实体列表，通常不会太大 |
| `edge_operations.py` | `extract_edges()` | 边抽取 | 输出是边列表，通常不会太大 |

### 低风险

| 文件 | 函数 | 风险描述 |
|-----|------|---------|
| `node_operations.py` | `extract_nodes_reflexion()` | 找遗漏实体，输出很小 |
| `node_operations.py` | `filter_extracted_nodes()` | 已实现分批处理（5实体/批） |
| `node_operations.py` | `_extract_entity_attributes()` | 单实体属性抽取 |
| `node_operations.py` | `_extract_entity_summary()` | 单实体摘要生成 |

---

## 已实施的修复

### 实体去重分批并行处理

**位置**: `graphiti_core/utils/maintenance/node_operations.py`

**配置参数**:
```python
DEDUP_BATCH_SIZE = 5      # 每个 LLM 请求处理的实体数量
DEDUP_PARALLELISM = 10    # 最大并发 LLM 请求数
```

**效果**:
| 场景 | 优化前 | 优化后 |
|-----|-------|-------|
| 27 个实体 | 1 次 LLM 调用，输出 55000+ 字符 | 6 次并行调用，每次 ~8000 字符 |
| 错误率 | JSON 截断导致失败 | 正常完成 |
| 总耗时 | N/A（失败） | 约等于单次调用（并行处理） |

**实现要点**:
1. 将实体列表按 `DEDUP_BATCH_SIZE` 分批
2. 每批内使用局部 ID (0 到 batch_size-1)
3. 调用 LLM 后将局部 ID 映射回全局索引
4. 使用 `semaphore_gather(max_coroutines=DEDUP_PARALLELISM)` 并行处理

---

## 监控建议

1. **日志监控**：关注 `JSONDecodeError` 或 `Unterminated string` 错误
2. **Token 监控**：如果使用的 LLM 支持，监控输出 token 使用量
3. **数据量监控**：单批次实体/边数量超过 20 时需要警惕

---

## 扩展方案

如果其他函数也出现类似问题，可参考以下模板：

```python
BATCH_SIZE = 5
PARALLELISM = 10

async def process_items_batched(items: list[T]) -> list[R]:
    # 分批
    batches = [items[i:i+BATCH_SIZE] for i in range(0, len(items), BATCH_SIZE)]

    async def process_batch(batch: list[T], batch_idx: int) -> list[tuple[int, R]]:
        # 构建局部上下文，ID 从 0 开始
        # 调用 LLM
        # 映射回全局索引
        ...

    # 并行处理
    results = await semaphore_gather(
        *[process_batch(b, i) for i, b in enumerate(batches)],
        max_coroutines=PARALLELISM,
    )

    # 合并结果
    return [r for batch_result in results for r in batch_result]
```

---

*文档更新时间: 2025-12-18*
*相关修复记录: easyops-customizations.md 第 20 节*
