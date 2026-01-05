# Cross-encoder Reranking 使用 name+summary（搜索质量修复）

## 问题背景

使用 Cross-encoder（如 Qwen Reranker）进行实体重排序时，原版 Graphiti 只使用 `node.name` 作为 passage 传给 cross_encoder.rank()。这导致 reranker 无法准确判断实体与查询的相关性。

**案例分析**：

对于查询 "What is the name of the father of Childericus?"：

| 实体名称 | 只用 name 的得分 | 用 name+summary 的得分 |
|---------|-----------------|----------------------|
| Childeric I | 0.06（排第9） | **0.95**（排第1） |
| Saints Maximus and Domitius | 0.45（排第1） | 0.22（排第4） |

只用 name 时，"Childeric I" 这个名字与问题中的 "father of Childericus" 看起来不相关（因为问的是父亲是谁，而不是 Childeric 本身）。但 summary 包含关键信息："son of Merovech"（Merovech 是正确答案）。

## 解决方案

**文件**: `graphiti_core/search/search.py`

修改 `node_search()` 函数中的 cross_encoder 处理逻辑：

```python
elif config.reranker == NodeReranker.cross_encoder:
    # EasyOps: 使用 name + summary 进行 cross_encoder reranking
    # 只用 name 会导致 reranker 无法判断相关性（例如问 "who is the father of X"，只有 summary 包含答案）
    text_to_uuid_map = {}
    for node in node_uuid_map.values():
        if node.summary:
            text = f"{node.name}: {node.summary}"
        else:
            text = node.name
        text_to_uuid_map[text] = node.uuid

    reranked_texts = await cross_encoder.rank(query, list(text_to_uuid_map.keys()))
    reranked_uuids = [
        text_to_uuid_map[text]
        for text, score in reranked_texts
        if score >= reranker_min_score
    ]
    node_scores = [score for _, score in reranked_texts if score >= reranker_min_score]
```

## 效果

修复后，使用 HotPotQA 数据集测试：

| 问题 | 修复前排名 | 修复后排名 |
|-----|-----------|-----------|
| Childeric I (父亲问题) | 第9 | **第1** |
| 其他多跳推理问题 | 经常丢失 | 正确排序 |

## 修改文件清单

| 文件 | 修改内容 |
|-----|---------|
| `graphiti_core/search/search.py` | Cross-encoder 使用 `name: summary` 格式作为 passage |

## 升级注意事项

此修改不影响已存储的数据，只影响搜索时的重排序逻辑。升级后无需重新导入数据。

---
