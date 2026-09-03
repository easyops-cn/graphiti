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
    # summary 为空的节点不参与 rerank：reranker 对裸名文本的判别不可靠，
    # 会让无 summary 的节点被错误顶高/压低；这些节点保留 RRF 检索顺序直接排在 rerank 结果之后
    text_to_uuid_map = {}
    nodes_without_summary: list[str] = []
    for node in node_uuid_map.values():
        if node.summary:
            text_to_uuid_map[f"{node.name}: {node.summary}"] = node.uuid
        else:
            nodes_without_summary.append(node.uuid)

    reranked_texts = await cross_encoder.rank(query, list(text_to_uuid_map.keys()))
    reranked_uuids = [
        text_to_uuid_map[text]
        for text, score in reranked_texts
        if score >= reranker_min_score
    ]
    node_scores = [score for _, score in reranked_texts if score >= reranker_min_score]
    # 无 summary 节点追加在 rerank 结果之后（保持检索顺序），不再送入 reranker
    reranked_uuids.extend(
        uuid for uuid in nodes_without_summary if uuid not in set(reranked_uuids)
    )
    node_scores.extend(0.0 for _ in range(len(reranked_uuids) - len(node_scores)))
```

### 空 summary 节点的处理（2026-09 复核：保持裸名送 rerank）

中间版本（2026-08）曾改为"summary 为空的节点不参与 rerank，追加在结果尾部"。
实测副作用：无 summary 的**目标本体**（域名实体，客户数据只有名字没有业务属性）
被 cosine 召回的有 summary 候选淹没，[:limit] 截断后完全丢失——域名精确搜索
3 例回退（`www.topsports.com.cn` 搜出一堆无关 pod）。

复核打分行为（2026-09，qwen3-reranker-0.6b via one.elevo.vip）：
- query == name 时稳定输出 0.99+（3 次复测波动 <0.1%），"裸名乱打分"不成立
- 真正的短板是细分度：目标 0.98 vs 词面重叠竞争者 0.986，分差决定排名

结论：维持裸名送 rerank（name 即 query 的实体天然高分）。细分度问题不在
reranker 解决，由 elevo 侧锚定层（identifier_pinning）用确定性规则处理：
身份属性精确等值 → 钉顶。

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
