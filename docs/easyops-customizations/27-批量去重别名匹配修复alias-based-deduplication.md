# 27. 批量去重别名匹配修复（Alias-Based Deduplication）

## 问题背景

批量导入时，Schema 中定义的别名实体（如 `EasyITSM 或 IT服务中心`）未被正确去重，导致同一实体创建多个节点。

**具体案例**：

ProductModule 类型定义包含别名模式：
```
(5)EasyITSM 或 ITSM 或 IT服务中心
```

预期：`EasyITSM`、`ITSM`、`IT服务中心` 应被识别为同一实体的不同名称。

实际：批量导入时这三个名称分别创建了独立节点。

## 根因分析

1. **`_resolve_batch_with_llm()` 未传递 `entity_type_definitions`**

   每个节点的 `entity_type_description` 包含别名信息，但 context 构建时没有传递 `entity_type_definitions`：

   ```python
   # 修改前 (bulk_utils.py:145-150)
   context = {
       'extracted_nodes': extracted_nodes_context,
       'existing_nodes': existing_nodes_context,
       'episode_content': '',
       'previous_episodes': [],
       # 缺失: 'entity_type_definitions' 未传递！
   }
   ```

2. **提示词未指导 LLM 使用别名信息**

   `dedupe_nodes.py` 的 `nodes()` 函数：
   - 期望 `entity_type_definitions` 但始终为空
   - 没有说明如何使用 `entity_type_description` 中的别名模式
   - 结果：LLM 只按字面名称匹配，不识别别名关系

## 解决方案

### 27.1 传递 entity_type_definitions

**文件**: `graphiti_core/utils/bulk_utils.py`

在 `_resolve_batch_with_llm()` 中构建并传递 `entity_type_definitions`：

```python
entity_types_dict = entity_types or {}

# Build entity_type_definitions for the prompt (EasyOps: enables alias-based deduplication)
# This tells LLM about type definitions including alias patterns like "EasyITSM 或 IT服务中心"
entity_type_definitions: dict[str, str] = {}
for node in nodes:
    for label in node.labels:
        if label != 'Entity' and label not in entity_type_definitions:
            type_model = entity_types_dict.get(label)
            if type_model and type_model.__doc__:
                entity_type_definitions[label] = type_model.__doc__

# ... later in context ...
context = {
    'extracted_nodes': extracted_nodes_context,
    'existing_nodes': existing_nodes_context,
    'episode_content': '',
    'previous_episodes': [],
    'entity_type_definitions': entity_type_definitions,  # EasyOps: enables alias detection
}
```

### 27.2 添加别名匹配指导

**文件**: `graphiti_core/prompts/dedupe_nodes.py`

在 `nodes()` 函数中添加别名匹配的明确指导：

```python
def nodes(context: dict[str, Any]) -> list[Message]:
    type_defs = context.get('entity_type_definitions', {})
    type_defs_section = ''
    alias_instruction = ''
    if type_defs:
        type_defs_section = f"""
        <ENTITY TYPE DEFINITIONS>
        {to_prompt_json(type_defs)}
        </ENTITY TYPE DEFINITIONS>
        """
        # EasyOps: Add explicit alias matching instruction when type definitions are available
        alias_instruction = """
        **IMPORTANT - ALIAS MATCHING**:
        The ENTITY TYPE DEFINITIONS above may contain alias patterns using "或" (Chinese "or").
        For example: "(5)EasyITSM 或 ITSM 或 IT服务中心" means EasyITSM, ITSM, and IT服务中心 are ALL aliases for the SAME entity.
        When you see entities with names matching these aliases, they MUST be marked as duplicates.
        """
```

同时更新实体结构描述，包含 `summary` 和 `entity_type_description` 字段：

```python
Each entity in ENTITIES is represented as a JSON object with the following structure:
{{
    id: integer id of the entity,
    name: "name of the entity",
    summary: "brief description of the entity",
    entity_type: ["Entity", "<optional additional label>", ...],
    entity_type_description: "description of the entity type, may contain alias patterns"
}}
```

## 效果

| 实体名称 | 修复前 | 修复后 |
|---------|-------|-------|
| EasyITSM | 独立节点 | 去重为同一节点 |
| IT服务中心 | 独立节点 | 去重为同一节点 |
| ITSM | 独立节点 | 去重为同一节点 |

LLM 现在能够：
1. 看到 `<ENTITY TYPE DEFINITIONS>` 中的类型定义和别名模式
2. 理解 "或" 表示的别名关系
3. 正确将不同名称但属于同一别名组的实体标记为重复

## 修改文件清单

| 文件 | 修改内容 |
|-----|---------|
| `graphiti_core/utils/bulk_utils.py` | `_resolve_batch_with_llm()` 构建并传递 `entity_type_definitions` |
| `graphiti_core/prompts/dedupe_nodes.py` | `nodes()` 添加 `alias_instruction` 和更完整的实体结构描述 |

---
