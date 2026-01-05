# 14. 两步类型验证优化（Two-Step Type Validation）

## 问题背景

原有的 `filter_entities` 提示词一次性传入：
- 所有 17+ 个 Schema 类型定义
- 长篇的 episode 原文内容
- 所有待验证的实体列表

这导致 LLM 注意力分散，无法准确判断每个实体是否符合其分配的类型定义。

**具体表现**：监控指标 ID（如 `system_cpu_cores`）被错误标记为 `Component`，但 LLM 在 filter 阶段看到 `Component` 是有效类型就直接保留了，没有验证该实体是否真正符合 Component 的定义（独立进程/服务/安装包）。

## 解决方案

将原来的单次 filter 调用拆分为两步：

**Step 1: 聚焦验证（Focused Validation）**
- 每个实体只传入**自己的**类型定义
- 不传入 episode 原文（只用 summary）
- 批量处理（5 个实体/批），并行执行
- 判断：实体是否符合其分配的类型定义？

**Step 2: 重新分类（Reclassification）**
- 只对 Step 1 中验证失败的实体
- 传入**所有**类型定义
- 判断：是否有其他类型匹配？还是应该删除？

## 实现细节

### 14.1 新增数据模型

**文件**: `graphiti_core/prompts/extract_nodes.py`

```python
# Step 1: 验证结果
class EntityValidationItem(BaseModel):
    name: str = Field(..., description='Name of the entity')
    is_valid: bool = Field(..., description='True if entity matches its assigned type definition')
    reason: str = Field(..., description='Brief explanation of why it matches or does not match')

class EntityValidationResult(BaseModel):
    validations: list[EntityValidationItem] = Field(..., description='Validation results for each entity')

# Step 2: 重新分类结果
class EntityReclassifyItem(BaseModel):
    name: str = Field(..., description='Name of the entity')
    new_type: str | None = Field(..., description='New type name if it matches another type, or null if should be removed')
    reason: str = Field(..., description='Brief explanation of the decision')

class EntityValidationResult(BaseModel):
    validations: list[EntityValidationItem] = Field(..., description='Validation results for each entity')
```

### 14.2 提示词函数

**Step 1 - `validate_entity_types(context)`**（新增）：
- 输入：实体列表，每个实体包含 `name`、`summary`、`assigned_type`、`type_definition`
- 不传入 episode 原文，只有实体自身信息
- 验证规则：检查 IS/IS NOT 示例，保守判断

**Step 2 - 复用 `extract_text(context)`**（生产验证过的提示词）：
- 输入：实体的 name+summary 作为"文本"，所有类型定义
- 让 LLM 重新抽取分类
- 如果分类为 Entity 或原类型 → 删除；否则重新分类

### 14.3 重写 filter_extracted_nodes

**文件**: `graphiti_core/utils/maintenance/node_operations.py`

```python
async def filter_extracted_nodes(...) -> tuple[list[str], list[EntityReclassification]]:
    # Step 1: Batch validation (5 entities per batch, parallel)
    async def validate_batch(batch):
        context = {'entities': batch}  # 每个实体只有自己的 type_definition
        return await llm_client.generate_response(
            prompt_library.extract_nodes.validate_entity_types(context),
            EntityValidationResult,
        )

    validation_results = await semaphore_gather(*[validate_batch(b) for b in batches])

    # Collect invalid entities
    invalid_entities = [e for e in validation_results if not e.is_valid]

    if not invalid_entities:
        return [], []

    # Step 2: Reclassify using production-validated extract_text prompt (parallel)
    async def reclassify_entity(entity):
        entity_text = f"Entity: {entity['name']}\nDescription: {entity['summary']}"
        context = {
            'episode_content': entity_text,
            'entity_types': entity_types_context,  # 所有类型定义
            # NOTE: Do NOT include validation_reason here - it may contain type suggestions
            # that would bias the LLM. Let extract_text make an independent classification.
            'custom_prompt': f"Classify the entity '{entity['name']}' based on its description. "
                           f"The entity was previously classified as '{entity['assigned_type']}' "
                           f"but that classification was incorrect. "
                           f"Please determine the correct entity type from the available types.",
        }
        return await llm_client.generate_response(
            prompt_library.extract_nodes.extract_text(context),  # 复用生产验证的提示词
            ExtractedEntities,
        )

    reclassify_results = await semaphore_gather(*[reclassify_entity(e) for e in invalid_entities])
    # ...
```

**重要**：Step 2 的 `custom_prompt` 不包含 Step 1 的 `validation_reason`，避免 LLM 被前一步的建议所偏向。

### 14.4 重新分类后更新 node.reasoning

**文件**: `graphiti_core/graphiti.py`

重新分类后不仅更新 `node.labels`，还要更新 `node.reasoning`，避免保留原类型的推理内容：

```python
# Apply reclassifications to nodes
if entities_to_reclassify:
    # Store both new_type and reason for updating node.reasoning
    reclassify_map = {r.name: (r.new_type, r.reason) for r in entities_to_reclassify}
    for node in hydrated_nodes:
        if node.name in reclassify_map:
            new_type, reason = reclassify_map[node.name]
            old_labels = node.labels.copy()
            node.labels = ['Entity', new_type] if new_type != 'Entity' else ['Entity']
            # Update reasoning with reclassification reason
            node.reasoning = f'[Reclassified from {old_labels} to {node.labels}] {reason}'
            logger.info(f'Reclassified "{node.name}": {old_labels} -> {node.labels}, reason: {reason}')
```

## 性能对比

| 指标 | 原方案 | 两步方案 |
|-----|-------|---------|
| LLM 调用次数 | 1 次（所有实体） | N/5 次（Step 1）+ M 次（Step 2，M=失败数） |
| 单次上下文大小 | 大（17 类型 + episode 原文） | 小（Step 1: 5实体×1类型; Step 2: 1实体×17类型） |
| 并行度 | 无 | **两步都并行** |
| 准确率 | 低（注意力分散） | 高（聚焦验证 + 复用生产验证提示词） |

## 效果

| 实体 | 原方案 | 两步方案 | 原因 |
|-----|-------|---------|------|
| `system_cpu_cores` (误标为 Component) | ✅ 保留（误） | ❌ Step 1 失败 → Step 2 无匹配 → 删除 | 不符合 Component 定义 |
| `监控套件` (正确标为 Feature) | ✅ 保留 | ✅ Step 1 通过 | 符合 Feature 定义 |
| `IT服务中心` (误标为 Entity) | ✅ 保留 | ✅ Step 1 失败 → Step 2 重分类为 ProductModule | 符合 ProductModule 定义 |

## 修改文件清单

| 文件 | 修改内容 |
|-----|---------|
| `graphiti_core/prompts/extract_nodes.py` | 新增 `EntityValidationItem`、`EntityValidationResult` 模型 |
| `graphiti_core/prompts/extract_nodes.py` | 新增 `validate_entity_types()` 提示词函数（Step 1） |
| `graphiti_core/prompts/extract_nodes.py` | 更新 `Versions` TypedDict 和 `versions` dict |
| `graphiti_core/utils/maintenance/node_operations.py` | 重写 `filter_extracted_nodes()` 为两步逻辑 |
| `graphiti_core/utils/maintenance/node_operations.py` | Step 2 的 `custom_prompt` 不包含 `validation_reason`（避免偏向） |
| `graphiti_core/utils/maintenance/node_operations.py` | 添加批量并行处理（5 实体/批）|
| `graphiti_core/graphiti.py` | 重新分类时同时更新 `node.reasoning`（`add_episode` 和 `_resolve_nodes_and_edges_bulk`） |

---
