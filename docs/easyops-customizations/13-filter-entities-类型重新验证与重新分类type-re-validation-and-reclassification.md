# 13. Filter Entities 类型重新验证与重新分类（Type Re-validation and Reclassification）

## 问题背景

`filter_entities` 步骤在检查实体是否应该保留时，会检查实体是否匹配 VALID ENTITY TYPES。但存在一个漏洞：

1. 实体在抽取阶段可能被错误分类（如监控指标 `system_cpu_cores` 被标记为 `Component`）
2. Filter 步骤看到实体有 `type: "Component"` 标签
3. LLM 看到 `Component` 是有效类型，就直接保留了
4. **没有验证实体是否真正符合 Component 的定义**

**具体案例**（trace_id: d1d3dcc90794422ee5c301af96fd6c62）：

```json
// 抽取阶段错误分类
{
  "name": "system_socket_summary_udp_all_count",
  "summary": "...",
  "type": "Component"  // 错误！这是监控指标，不是组件
}

// Filter 阶段 LLM 的错误推理
{
  "entities_to_remove": [],
  "reasoning": "All extracted entities represent system monitoring metrics
   that are core components of the EasyOps monitoring system..."
}
```

Component 的定义要求：
> 有独立的进程/服务/安装包？运维可以独立启停、配置、监控？

但 `system_socket_summary_udp_all_count` 是一个**监控指标 ID**，不符合 Component 定义。

## 解决方案

### 13.1 新增类型重新分类功能

**原有设计**：误分类的实体只能被删除，可能丢失有价值的实体。

**新设计**：误分类的实体可以被重新分类到正确的类型，保留有价值的实体。

**新增模型** (`graphiti_core/prompts/extract_nodes.py`)：

```python
class EntityReclassification(BaseModel):
    name: str = Field(..., description='Name of the entity to reclassify')
    new_type: str = Field(
        ...,
        description='The correct entity type name from VALID ENTITY TYPES. Must be an exact match.',
    )
    reason: str = Field(..., description='Brief explanation of why this type is more appropriate')


class EntitiesToFilter(BaseModel):
    entities_to_remove: list[str] = Field(...)
    entities_to_reclassify: list[EntityReclassification] = Field(  # 新增
        default_factory=list,
        description='Entities that were misclassified and should be assigned a different type',
    )
    reasoning: str = Field(...)
```

### 13.2 修改 Filter 提示词

**System Prompt 增加 CRITICAL 提示**：

```python
sys_prompt = """You are a knowledge graph quality reviewer...

**CRITICAL**: Entities may have a pre-assigned "type" from the extraction step.
Do NOT trust this type blindly. You MUST re-validate the entity against the
type's definition in VALID ENTITY TYPES. If the entity does not actually match
its assigned type's criteria (especially the IS/IS NOT examples in the type description):
- If it matches a DIFFERENT valid type, RECLASSIFY it
- If it doesn't match ANY valid type, REMOVE it"""
```

**Decision Process 改为类型重新验证 + 重新分类**：

```python
**Decision Process**:
1. If the entity has a pre-assigned type, RE-VALIDATE it against that type's definition:
   - Check if it matches the type's criteria (judgment standards)
   - Check if it matches the IS examples (should be kept with current type)
   - Check if it matches the IS NOT examples (needs reclassification or removal)
   - If MISCLASSIFIED but matches a DIFFERENT valid type → add to entities_to_reclassify
   - If MISCLASSIFIED and doesn't match ANY valid type → add to entities_to_remove
2. If the entity has no type or type is "Entity", check if it matches any valid type:
   - If it matches a valid type → add to entities_to_reclassify with the correct type
   - If it doesn't match any type, apply the four principles strictly
```

**添加常见误分类提示**：

```python
**Common Misclassifications to Watch For**:
- Metric IDs (like system_cpu_cores, system_memory_total) are NOT Components - Components are deployable services
- Table columns or field names are NOT CmdbModels - CmdbModels are model definitions
- Generic technical terms are NOT Features - Features have menu entries in the product
```

### 13.3 修改 filter_extracted_nodes 函数

**文件**: `graphiti_core/utils/maintenance/node_operations.py`

返回类型从 `list[str]` 改为 `tuple[list[str], list[EntityReclassification]]`：

```python
async def filter_extracted_nodes(
    llm_client: LLMClient,
    episode: EpisodicNode,
    nodes: list[EntityNode],
    group_id: str | None = None,
    entity_types_context: list[dict] | None = None,
) -> tuple[list[str], list[EntityReclassification]]:
    """Filter out entities and identify misclassified entities for reclassification.

    Returns:
        Tuple of (entities_to_remove, entities_to_reclassify)
    """
```

### 13.4 修改调用方处理重新分类

**文件**: `graphiti_core/graphiti.py`

**单个 Episode 处理**（`add_episode` 方法）：

```python
entities_to_remove, entities_to_reclassify = await filter_extracted_nodes(...)

# Apply reclassifications to nodes
if entities_to_reclassify:
    reclassify_map = {r.name: r.new_type for r in entities_to_reclassify}
    for node in hydrated_nodes:
        if node.name in reclassify_map:
            new_type = reclassify_map[node.name]
            old_labels = node.labels.copy()
            node.labels = ['Entity', new_type] if new_type != 'Entity' else ['Entity']
            logger.info(f'Reclassified "{node.name}": {old_labels} -> {node.labels}')
```

**批量 Episode 处理**（`_resolve_nodes_and_edges_bulk` 方法）：

```python
# Collect all reclassifications
all_reclassifications: dict[str, str] = {}  # name -> new_type
for entities_to_remove, entities_to_reclassify in filter_results:
    for reclass in entities_to_reclassify:
        all_reclassifications[reclass.name] = reclass.new_type

# Apply reclassifications
if all_reclassifications:
    for nodes in hydrated_nodes_results:
        for node in nodes:
            if node.name in all_reclassifications:
                new_type = all_reclassifications[node.name]
                node.labels = ['Entity', new_type] if new_type != 'Entity' else ['Entity']
```

## 效果

| 实体 | 修改前 | 修改后 | 原因 |
|-----|-------|-------|------|
| `system_cpu_cores` (误标为 Component) | ✅ 保留（误） | ❌ 移除 | 不符合任何有效类型 |
| `IT服务中心` (误标为 Entity) | ✅ 保留但类型错误 | ✅ 重新分类为 ProductModule | 符合 ProductModule 定义 |
| `cmdb_service` | ✅ 保留 | ✅ 保留 | 符合 Component 定义（独立服务） |

## 修改文件清单

| 文件 | 修改内容 |
|-----|---------|
| `graphiti_core/prompts/extract_nodes.py` | 新增 `EntityReclassification` 模型 |
| `graphiti_core/prompts/extract_nodes.py` | `EntitiesToFilter` 添加 `entities_to_reclassify` 字段 |
| `graphiti_core/prompts/extract_nodes.py` | `filter_entities()` 提示词支持重新分类 |
| `graphiti_core/utils/maintenance/node_operations.py` | `filter_extracted_nodes()` 返回类型改为 tuple |
| `graphiti_core/graphiti.py` | `add_episode()` 处理重新分类 |
| `graphiti_core/graphiti.py` | `_resolve_nodes_and_edges_bulk()` 处理重新分类 |

---
