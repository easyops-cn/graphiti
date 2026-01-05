# 5. 批量去重 LLM 语义去重（Semantic Dedup）

## 问题背景

批量导入时（`add_episode_bulk`），同一批次内的实体去重存在问题：

1. **确定性匹配局限**：精确字符串匹配和 MinHash Jaccard ≥ 0.9 无法处理语义同义词
2. **中英文同义词**：如 `EasyITSM` vs `IT服务中心` vs `EasyITSC`（同一产品的不同名称）
3. **Summary 时机问题**：原有的批量去重在 `extract_attributes_from_nodes` 之前执行，此时 summary 为空

导致同一实体被创建为多个节点。

## 解决方案

**文件**:
- `graphiti_core/utils/bulk_utils.py` - 新增函数
- `graphiti_core/graphiti.py` - 调用入口

在 `extract_attributes_from_nodes` 之后执行 LLM 语义去重，此时 summary 和 attributes 已填充。

### 5.1 新增辅助函数

```python
def _get_entity_type_label(node: EntityNode) -> str:
    """获取实体的具体类型标签（非 'Entity'）"""

async def _resolve_batch_with_llm(...) -> list[tuple[EntityNode, EntityNode]]:
    """调用 LLM 识别语义重复，返回 (source, canonical) 节��对"""

def _merge_node_into_canonical(source: EntityNode, canonical: EntityNode) -> None:
    """合并 source 的 summary 和 attributes 到 canonical"""
```

### 5.2 新增主函数

```python
async def semantic_dedupe_nodes_bulk(
    clients: GraphitiClients,
    nodes: list[EntityNode],
    entity_types: dict[str, type[BaseModel]] | None = None,
) -> list[EntityNode]:
    """
    在 extract_attributes_from_nodes 之后执行 LLM 语义去重。
    按实体类型分组，对同类型实体调用 LLM 检查是否为语义重复。
    发现重复时合并 summary 和 attributes。
    """
```

### 5.3 调用位置

```python
# graphiti.py add_episode_bulk 中
(final_hydrated_nodes, ...) = await self._resolve_nodes_and_edges_bulk(...)

# EasyOps: 在 attributes 提取后执行语义去重
final_hydrated_nodes = await semantic_dedupe_nodes_bulk(
    self.clients, final_hydrated_nodes, entity_types
)

# 保存到图数据库
await add_nodes_and_edges_bulk(...)
```

## 合并逻辑

发现重复时，将 source 节点的信息合并到 canonical 节点：

```python
def _merge_node_into_canonical(source, canonical):
    # Summary: 拼接（如果都有且不重复）
    if source.summary and canonical.summary:
        if source.summary not in canonical.summary:
            canonical.summary = f"{canonical.summary} {source.summary}"
    elif source.summary:
        canonical.summary = source.summary

    # Attributes: source 填充 canonical 缺失的字段
    for key, value in source.attributes.items():
        if key not in canonical.attributes or not canonical.attributes[key]:
            canonical.attributes[key] = value
```

## 日志输出

- `[semantic_dedup] Checking X ProductModule entities for semantic duplicates`
- `[batch_dedup_llm] Sending 1 nodes to LLM against Y candidates`
- `[batch_dedup_llm] Duplicate found: "EasyITSM" -> "IT服务中心"`
- `[semantic_dedup] Merged "EasyITSM" into "IT服务中心"`

## 效果

- 批量导入时能正确识别语义同义词
- 有 summary 信息帮助 LLM 判断
- 合并重复实体的知识，不丢失信息

## 合并逻辑（_merge_node_into_canonical）

合并时会合并以下字段：
- **summary**: 拼接（如果都有且不重复）
- **attributes**: source 填充 canonical 缺失的字段
- **reasoning**: 拼接（用 `\n---\n` 分隔，保留所有推理过程）

---
