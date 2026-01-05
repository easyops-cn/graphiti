# 6. 边去重时合并 Fact（Edge Fact Merging）

## 问题背景

原始 Graphiti 的边去重逻辑存在问题：
1. 当检测到重复边时，直接丢弃新边的 fact，只保留旧边的 fact
2. 不同语言或不同表述的相同关系没有被识别为重复
3. 重复边的属性（valid_at, invalid_at, attributes）没有合并
4. **关键问题**：同一对节点之间、同一边类型的边，即使 fact 描述不同，也会创建多条重复边

## 解决方案

### 7.1 添加 merged_fact 字段

**文件**: `graphiti_core/prompts/dedupe_edges.py`

在 `EdgeDuplicate` 模型中添加 `merged_fact` 字段：

```python
class EdgeDuplicate(BaseModel):
    duplicate_facts: list[int] = Field(...)
    merged_fact: str | None = Field(
        default=None,
        description='When duplicate_facts is not empty, provide a merged fact that combines '
        'the semantic meaning of both the NEW FACT and the EXISTING FACT(s)...',
    )
    contradicted_facts: list[int] = Field(...)
    fact_type: str = Field(...)
    fact_type_reasoning: str = Field(...)
```

### 7.2 更新提示词

**文件**: `graphiti_core/prompts/dedupe_edges.py`

在 `resolve_edge` 提示词中增加：

1. **重复检测增强**：明确"不同语言、不同措辞表达相同关系"也是重复
2. **Fact 合并任务**：当检测到重复时，LLM 必须返回合并后的 fact
3. **合并指南**：
   - 合并两个 fact 的语义信息
   - 去除冗余
   - 如果是不同语言，选择更详细或更一致的语言

### 7.3 确定性边去重（2025-12-12 优化）

**核心改进**：边去重从"LLM 判断是否重复"改为"程序确定性判断 + LLM 合并 fact"。

**原理**：`(source_uuid, target_uuid, edge_type)` 相同的边就是同一条边，这是确定性逻辑，不需要 LLM 判断。LLM 只负责合并不同表述的 fact。

**文件**: `graphiti_core/utils/maintenance/edge_operations.py`

**批内去重** - `resolve_extracted_edges()` 函数：

```python
# 原来：基于 (source, target, fact) 去重
key = (
    edge.source_node_uuid,
    edge.target_node_uuid,
    _normalize_string_exact(edge.fact),  # fact 不同就认为是不同的边
)

# 修改后：基于 (source, target, edge_type) 去重
key = (
    edge.source_node_uuid,
    edge.target_node_uuid,
    edge.name,  # edge_type 相同就是同一条边
)
if key not in seen:
    seen[key] = edge
    facts_to_merge[edge.uuid] = [edge.fact]
else:
    # 同一对节点的同类型边，收集 fact 待合并
    canonical_edge = seen[key]
    facts_to_merge[canonical_edge.uuid].append(edge.fact)
    # 合并 episode 引用
    for ep_uuid in edge.episodes:
        if ep_uuid not in canonical_edge.episodes:
            canonical_edge.episodes.append(ep_uuid)
```

**与数据库已有边去重** - `resolve_extracted_edge()` 函数的快速路径：

```python
# 原来：基于 (source, target, fact) 精确匹配
if _normalize_string_exact(edge.fact) == normalized_fact:
    return resolved, [], []

# 修改后：基于 (source, target, edge_type) 匹配
if edge.name == extracted_edge.name:  # 同边类型 = 重复
    resolved = edge
    resolved.episodes.append(episode.uuid)
    return resolved, [extracted_edge], []  # 标记为重复，调用方处理 fact 合并
```

**效果**：
- 同一对节点间、同一边类型的边，无论 fact 如何不同，都会被识别为同一条边
- 多条 fact 描述会被合并（LLM 合并或拼接）
- 避免创建语义重复的边

### 7.4 修改合并逻辑

**文件**: `graphiti_core/utils/maintenance/edge_operations.py`

修改 `resolve_extracted_edge` 函数中的合并逻辑，支持两种情况：

```python
for duplicate_fact_id in duplicate_fact_ids:
    existing_edge = related_edges[duplicate_fact_id]
    merged_fact = response_object.merged_fact
    if merged_fact and merged_fact.strip():
        # 使用 LLM 返回的合并 fact
        existing_edge.fact = merged_fact.strip()
        existing_edge.fact_embedding = None
        logger.info(f'[edge_dedup] Updated fact for edge {existing_edge.uuid} with LLM-merged content')
    else:
        # 强制合并场景：LLM 没有返回 merged_fact，手动拼接
        new_fact = extracted_edge.fact.strip()
        existing_fact = existing_edge.fact.strip()
        if new_fact and new_fact not in existing_fact and existing_fact not in new_fact:
            existing_edge.fact = f'{existing_fact} {new_fact}'
            existing_edge.fact_embedding = None
            logger.info(f'[edge_dedup] Concatenated fact for edge {existing_edge.uuid}')
    # 合并属性、valid_at、invalid_at（同原逻辑）
    ...
```

## 合并策略

| 字段 | 合并策略 |
|-----|---------|
| `fact` | 优先使用 LLM 合并结果；如无则拼接（去重后） |
| `fact_embedding` | 清空，由后续流程重新计算 |
| `episodes` | 追加新 episode UUID（原有逻辑） |
| `attributes` | 新边补充旧边缺失的字段 |
| `valid_at` | 取较早的时间戳 |
| `invalid_at` | 取较晚的时间戳 |

## 修改文件清单

| 文件 | 修改内容 |
|-----|---------|
| `graphiti_core/prompts/dedupe_edges.py` | `EdgeDuplicate` 添加 `merged_fact` 字段 |
| `graphiti_core/prompts/dedupe_edges.py` | `resolve_edge` 提示词增加合并任务 |
| `graphiti_core/utils/maintenance/edge_operations.py` | `resolve_extracted_edge` 强制合并同一对节点的边 |
| `graphiti_core/utils/maintenance/edge_operations.py` | `resolve_extracted_edge` 支持拼接 fact（无 LLM 合并时） |

---
