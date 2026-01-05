# 两轮 Map-Reduce 语义去重（Semantic Dedup with Map-Reduce）

## 问题背景

之前的去重流程存在问题：

1. `resolve_extracted_nodes` 在 `_resolve_nodes_and_edges_bulk` 中被调用
2. 此时节点还没有 summary，LLM 只能看到 name，决策质量差
3. 是串行执行的（一个 episode 一个 episode 处理）

## 解决方案

1. **移除** `_resolve_nodes_and_edges_bulk` 中的 `resolve_extracted_nodes` 调用
2. **启用** `semantic_dedupe_nodes_bulk`，在 `extract_attributes_from_nodes` 之后执行
3. **改为两轮 Map-Reduce**：分批并行 + 跨批去重

## 修改后的流程

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

## Map-Reduce 设计

### 第一轮：Map（分批并行去重）

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

### 第二轮：Reduce（跨批去重）

```
输入: 10 个批次的代表节点
     ↓
跨批去重: LLM 判断不同批次的节点是否重复
     ↓
输出: 最终的 uuid_map（使用并查集合并）
```

## 新增函数

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

## 冲突处理

使用并查集处理：
- `A->B, B->C` 折叠为 `A->C, B->C`
- `A->B, A->C` 保留第一个（先到先得）

## 配置参数

- `batch_size`: 每批节点数，默认 10

## 修改文件清单

| 文件 | 修改内容 |
|-----|---------|
| `graphiti_core/graphiti.py` | 删除 `resolve_extracted_nodes` 调用 |
| `graphiti_core/graphiti.py` | 启用 `semantic_dedupe_nodes_bulk` 调用 |
| `graphiti_core/utils/bulk_utils.py` | 重写 `semantic_dedupe_nodes_bulk` 为 Map-Reduce |
| `graphiti_core/utils/bulk_utils.py` | 新增 `_map_batch_dedup()` 函数 |
| `graphiti_core/utils/bulk_utils.py` | 新增 `_reduce_cross_batch_dedup()` 函数 |
| `graphiti_core/utils/bulk_utils.py` | 新增 `_build_uuid_map_from_pairs()` 函数 |

---
