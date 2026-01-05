# 10. 批量语义去重性能优化（Semantic Dedup Batch Optimization）- **已回滚**

## 问题背景

`add_episode_bulk` 批量导入时，`semantic_dedupe_nodes_bulk` 函数对同类型实体进行 O(n²) 的串行 LLM 调用：

```python
# 原实现：54 个 Feature 实体 → 54 次串行 LLM 调用
for i, node in enumerate(type_nodes):
    candidates = [n for n in type_nodes[i+1:] if n.uuid not in duplicate_map]
    llm_pairs = await _resolve_batch_with_llm([node], candidates, entity_types)
```

性能分析显示：54 个 Feature 实体的语义去重耗时 214 秒（占总时间 42%）。

## 曾经的优化方案（已回滚）

曾将 O(n²) 串行调用改为 O(entity_type_count) 批量调用，每种实体类型只需 1 次 LLM 调用。

## 回滚原因（2025-12-15）

批量自比较的实现有逻辑缺陷：把所有节点同时作为 ENTITIES 和 EXISTING ENTITIES，导致 LLM 每个节点都能在 EXISTING 中找到自己（100% 匹配），从而不会去比较其他节点是否重复。

**具体表现**：`ITSC管理平台` 和 `EasyITSM` 虽然 `code="ITSC"` 相同且 summary 功能描述高度相似，但 LLM 各自匹配到自己，没有识别为重复。

## 当前实现

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

## 修改文件清单

| 文件 | 修改内容 |
|-----|---------|
| `graphiti_core/utils/bulk_utils.py` | `_resolve_batch_self_dedup()` → `_resolve_batch_with_llm()` |
| `graphiti_core/utils/bulk_utils.py` | `semantic_dedupe_nodes_bulk()` 回滚到 O(n²) 实现 |

---
