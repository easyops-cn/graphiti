# 实体去重 LLM 调用优化（Dedup LLM Call Optimization）

## 问题背景

原有的 Map-Reduce 去重存在严重的性能问题：

1. **全局 DB 候选收集**：在去重前收集所有节点的 DB 候选（每节点搜 20 个），导致大量 DB 候选被收集但实际只用一部分
2. **O(n²) 批内去重**：每个节点单独调用 LLM 对比后续节点，N 个节点需要 N 次 LLM 调用
3. **批内去重 + DB 匹配分离**：分两步处理，增加了 LLM 调用次数

**实际案例**：91 个 Feature 实体产生了 186 次 LLM 调用（理论上应该远少于此）。

## 解决方案

### 1. 每批独立搜索 DB 候选

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

### 2. 合并批内去重 + DB 匹配为单次 LLM 调用

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

### 3. Reduce 阶段简化

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

### 4. 移除全局 DB 候选收集

`dedupe_nodes_bulk` 不再返回 `db_candidates_by_type`，函数签名从：
```python
async def dedupe_nodes_bulk(...) -> tuple[dict, dict, dict]:
```
改为：
```python
async def dedupe_nodes_bulk(...) -> tuple[dict, dict]:
```

## 优化效果

| 指标 | 修改前 | 修改后 |
|-----|-------|-------|
| Map 阶段 LLM 调用 | N × batch_count 次 | batch_count 次 |
| Reduce 阶段 LLM 调用 | N × (N-1)/2 次 | 1 次 |
| 每节点 DB 候选数 | 20 | 10 |
| 候选收集时机 | 全局预收集 | 按批实时搜索 |

**预期效果**：91 个 Feature 实体从 186 次 LLM 调用减少到约 10 次（10 个 batch × 1 次 + 1 次 Reduce）。

## 修改文件清单

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
