# 7. 实体去重 Reasoning 字段与 Prompt 优化

## 问题背景

实体去重时 LLM 做出错误判断（将不同实体判定为重复），但没有解释原因，难以调试和定位问题。

**具体案例**：`监控套件` 被错误添加了同义词 `SNMP监控套件`、`业务墙`、`拨测详情`。

## 解决方案

### 8.1 添加 reasoning 字段

**文件**: `graphiti_core/prompts/dedupe_nodes.py`

在 `NodeDuplicate` 模型中添加 `reasoning` 字段：

```python
class NodeDuplicate(BaseModel):
    id: int = Field(...)
    duplicate_idx: int = Field(...)
    name: str = Field(...)
    duplicates: list[int] = Field(...)
    reasoning: str = Field(
        default='',
        description='Brief explanation of why this entity is or is not a duplicate. Required when duplicate_idx != -1.',
    )
```

### 8.2 更新 Prompt 要求输出 reasoning

**文件**: `graphiti_core/prompts/dedupe_nodes.py`

修改 `nodes()` 和 `node()` 函数的 prompt，在输出格式中添加 reasoning 字段：

```python
For every entity, return an object with the following keys:
{{
    "id": integer id from ENTITIES,
    "name": the best full name for the entity,
    "duplicate_idx": the idx of the EXISTING ENTITY that is the best duplicate match, or -1 if there is no duplicate,
    "duplicates": a sorted list of all idx values from EXISTING ENTITIES that refer to duplicates,
    "reasoning": a brief explanation (1-2 sentences) of why you determined this entity is or is not a duplicate. REQUIRED when duplicate_idx != -1.
}}

- When marking as duplicate, explain what evidence shows they refer to the same real-world object.
- When NOT marking as duplicate, you may leave reasoning empty or briefly explain why they are distinct.
```

### 8.3 优化 entity_type_definitions（避免重复）

**问题**：每个实体的 `entity_type_description` 完全重复，浪费 token。

**解决方案**：将 entity_type_definitions 独立为一个部分，在实体列表中只保留类型名称。

**文件**: `graphiti_core/utils/maintenance/node_operations.py`

```python
# 修改前：每个实体都带 entity_type_description
extracted_nodes_context = [
    {
        'id': i,
        'name': node.name,
        'entity_type': node.labels,
        'entity_type_description': entity_types_dict.get(...).__doc__,  # 重复！
    }
    for i, node in enumerate(llm_extracted_nodes)
]

# 修改后：独立提取类型定义
entity_type_definitions: dict[str, str] = {}
for node in llm_extracted_nodes:
    for label in node.labels:
        if label != 'Entity' and label not in entity_type_definitions:
            type_model = entity_types_dict.get(label)
            if type_model and type_model.__doc__:
                entity_type_definitions[label] = type_model.__doc__

extracted_nodes_context = [
    {
        'id': i,
        'name': node.name,
        'entity_type': node.labels,  # 只有类型名称，不重复描述
    }
    for i, node in enumerate(llm_extracted_nodes)
]

context = {
    'extracted_nodes': extracted_nodes_context,
    'existing_nodes': existing_nodes_context,
    'entity_type_definitions': entity_type_definitions,  # 独立部分
    ...
}
```

**文件**: `graphiti_core/prompts/dedupe_nodes.py`

修改 `nodes()` 函数，添加 `ENTITY TYPE DEFINITIONS` 部分：

```python
def nodes(context: dict[str, Any]) -> list[Message]:
    # Build entity type definitions section if available
    type_defs = context.get('entity_type_definitions', {})
    type_defs_section = ''
    if type_defs:
        type_defs_section = f"""
        <ENTITY TYPE DEFINITIONS>
        {to_prompt_json(type_defs)}
        </ENTITY TYPE DEFINITIONS>
        """

    return [
        Message(
            role='user',
            content=f"""
        ...
        {type_defs_section}

        Each entity in ENTITIES is represented as a JSON object with the following structure:
        {{
            id: integer id of the entity,
            name: "name of the entity",
            entity_type: ["Entity", "<optional additional label>", ...]
        }}
        ...
        """
        ),
    ]
```

### 8.4 添加日志记录

**文件**: `graphiti_core/utils/maintenance/node_operations.py`

在 `_resolve_with_llm()` 函数中，判定为重复时记录 reasoning：

```python
if resolved_node.uuid != extracted_node.uuid:
    state.duplicate_pairs.append((extracted_node, resolved_node))
    # Log deduplication decision with reasoning for debugging
    logger.info(
        'Dedupe: "%s" -> "%s" (reasoning: %s)',
        extracted_node.name,
        resolved_node.name,
        resolution.reasoning or 'no reasoning provided',
    )
```

## 效果

1. **调试能力增强**：可以从日志中看到 LLM 判定重复的原因
2. **Token 节省**：entity_type_definitions 不再重复，每种类型只出现一次
3. **Prompt 结构清晰**：类型定义独立，实体列表更简洁

## 修改文件清单

| 文件 | 修改内容 |
|-----|---------|
| `graphiti_core/prompts/dedupe_nodes.py` | `NodeDuplicate` 添加 `reasoning` 字段 |
| `graphiti_core/prompts/dedupe_nodes.py` | `nodes()` 和 `node()` prompt 要求输出 reasoning |
| `graphiti_core/prompts/dedupe_nodes.py` | `nodes()` 添加 `ENTITY TYPE DEFINITIONS` 独立部分 |
| `graphiti_core/utils/maintenance/node_operations.py` | `_resolve_with_llm()` 独立提取 entity_type_definitions |
| `graphiti_core/utils/maintenance/node_operations.py` | `_resolve_with_llm()` 添加 reasoning 日志 |

---
