# 31. RRF 搜索配置启用 BFS（Enable BFS in RRF Search Config）

## 问题背景

Graphiti 提供多种预定义的搜索配置（Search Config Recipes），决定搜索时使用哪些检索方法：

| 配置 | 边检索方法 | 节点检索方法 |
|-----|-----------|-------------|
| `COMBINED_HYBRID_SEARCH_RRF` | BM25, Cosine | BM25, Cosine |
| `COMBINED_HYBRID_SEARCH_CROSS_ENCODER` | BM25, Cosine, **BFS** | BM25, Cosine, **BFS** |

Elevo Memory 的搜索逻辑：

```python
# search.py
if settings.qwen_reranker_enabled:
    base_config = COMBINED_HYBRID_SEARCH_CROSS_ENCODER  # 包含 BFS
elif settings.search_enable_llm_reranker:
    base_config = COMBINED_HYBRID_SEARCH_CROSS_ENCODER  # 包含 BFS
else:
    base_config = COMBINED_HYBRID_SEARCH_RRF  # 不包含 BFS！
```

**问题**：当 reranker 未启用时（默认配置），使用 `COMBINED_HYBRID_SEARCH_RRF`，该配置**不包含 BFS**，导致无法通过图遍历发现关联实体。

第 30 章节修复了 BFS 的边类型限制问题，但如果搜索配置根本不调用 BFS，修复就无法生效。

## 解决方案

在 `COMBINED_HYBRID_SEARCH_RRF` 中添加 BFS 搜索方法：

**文件**: `graphiti_core/search/search_config_recipes.py`

```python
# 修改前
COMBINED_HYBRID_SEARCH_RRF = SearchConfig(
    edge_config=EdgeSearchConfig(
        search_methods=[EdgeSearchMethod.bm25, EdgeSearchMethod.cosine_similarity],
        reranker=EdgeReranker.rrf,
    ),
    node_config=NodeSearchConfig(
        search_methods=[NodeSearchMethod.bm25, NodeSearchMethod.cosine_similarity],
        reranker=NodeReranker.rrf,
    ),
    # ...
)

# 修改后
COMBINED_HYBRID_SEARCH_RRF = SearchConfig(
    edge_config=EdgeSearchConfig(
        search_methods=[EdgeSearchMethod.bm25, EdgeSearchMethod.cosine_similarity, EdgeSearchMethod.bfs],
        reranker=EdgeReranker.rrf,
    ),
    node_config=NodeSearchConfig(
        search_methods=[NodeSearchMethod.bm25, NodeSearchMethod.cosine_similarity, NodeSearchMethod.bfs],
        reranker=NodeReranker.rrf,
    ),
    # ...
)
```

## 效果

| 场景 | 修复前 | 修复后 |
|-----|-------|-------|
| 无 reranker 时的多跳检索 | ❌ 不触发 BFS | ✅ 正常 BFS 遍历 |
| 边向量 → 边 BFS 关联 | ❌ 仅向量检索 | ✅ 向量 + BFS |
| 节点向量 → 节点 BFS 关联 | ❌ 仅向量检索 | ✅ 向量 + BFS |

## 与第 30 章节的关系

- **第 30 章节**：修复 BFS 的边类型限制（从 `RELATES_TO|MENTIONS` 改为遍历所有边类型）
- **第 31 章节**：确保 BFS 被调用（在 RRF 配置中启用 BFS）

两个修复缺一不可：
1. 如果只有第 30 章节，但 RRF 配置不调用 BFS，修复无效
2. 如果只有第 31 章节，但 BFS 只遍历 `RELATES_TO|MENTIONS`，动态边类型仍无法发现

## 修改文件清单

| 文件 | 修改内容 |
|-----|---------|
| `graphiti_core/search/search_config_recipes.py` | `COMBINED_HYBRID_SEARCH_RRF` 的 edge_config 和 node_config 添加 BFS |

---
