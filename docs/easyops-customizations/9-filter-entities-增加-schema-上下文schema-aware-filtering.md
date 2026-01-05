# 9. Filter Entities 增加 Schema 上下文（Schema-Aware Filtering）

## 问题背景

`filter_entities` 步骤原本只应用四大原则过滤实体，但没有领域 Schema 知识：

1. **误删合法实体**：`IT服务中心`（ProductModule）被误删，因为 LLM 认为它是"UI navigation"
2. **误删功能名称**：`服务目录`（Feature）被误删，因为 LLM 认为它是"document-specific section title"

**根因**：filter 步骤没有拿到 Schema 实体类型定义，不知道哪些是合法的领域概念。

## 解决方案

让 filter 步骤拿到完整的 Schema 实体类型定义，在应用四大原则前先检查实体是否匹配已定义的类型。

### 9.1 修改 filter_entities 提示词

**文件**: `graphiti_core/prompts/extract_nodes.py`

```python
def filter_entities(context: dict[str, Any]) -> list[Message]:
    sys_prompt = """You are a knowledge graph quality reviewer...

**IMPORTANT**: Before removing an entity, check if it matches any of the VALID ENTITY TYPES defined in the schema. If an entity clearly belongs to a defined type (based on the type's description and examples), it should be KEPT even if it seems document-specific."""

    # Build entity types reference if available
    entity_types_ref = ''
    if context.get('entity_types'):
        entity_types_ref = '\n<VALID ENTITY TYPES>\n'
        for et in context['entity_types']:
            if et.get('entity_type_name') != 'Entity':  # Skip default type
                entity_types_ref += f"- {et.get('entity_type_name')}: {et.get('entity_type_description', '')}\n"
        entity_types_ref += '</VALID ENTITY TYPES>\n'

    user_prompt = f"""
...
{entity_types_ref}
Review each extracted entity. Return the names of entities that FAIL the Knowledge Graph Builder's Principles.

**Decision Process**:
1. First, check if the entity matches any VALID ENTITY TYPE (if provided). If it clearly fits a defined type based on the type's description and examples, KEEP it.
2. Only if the entity doesn't match any valid type, apply the four principles strictly.
...
"""
```

### 9.2 修改 filter_extracted_nodes 函数

**文件**: `graphiti_core/utils/maintenance/node_operations.py`

添加 `entity_types_context` 参数：

```python
async def filter_extracted_nodes(
    llm_client: LLMClient,
    episode: EpisodicNode,
    extracted_entities: list[ExtractedEntity],
    group_id: str | None = None,
    entity_types_context: list[dict] | None = None,  # 新增
) -> list[str]:
    """Filter out entities that don't meet knowledge graph quality standards.

    If entity_types_context is provided, entities matching valid types will be preserved.
    """
    if not extracted_entities:
        return []

    context = {
        'episode_content': episode.content,
        'extracted_entities': [e.name for e in extracted_entities],
        'entity_types': entity_types_context,  # 传入 Schema 类型定义
    }
    # ...
```

### 9.3 修改调用方

**文件**: `graphiti_core/utils/maintenance/node_operations.py`

在 `extract_nodes()` 函数中传入 `entity_types_context`：

```python
# Filter entities using Knowledge Graph Builder's Principles
entities_to_remove = await filter_extracted_nodes(
    llm_client,
    episode,
    extracted_entities,
    episode.group_id,
    entity_types_context,  # 新增：传入实体类型上下文
)
```

## 决策流程

修改后的 filter 决策流程：

```
对每个待过滤的实体：
  1. 检查是否匹配 Schema 定义的类型
     - 匹配 → KEEP（即使看起来像文档元素）
     - 不匹配 → 进入下一步
  2. 应用四大原则
     - 通过全部 → KEEP
     - 任一失败 → REMOVE
```

## 效果

| 实体 | 修改前 | 修改后 | 原因 |
|-----|-------|-------|------|
| IT服务中心 | ❌ 误删 | ✅ 保留 | 匹配 ProductModule 定义 |
| 服务目录 | ❌ 误删 | ✅ 保留 | 匹配 Feature 定义 |
| 流程编排 | ❌ 误删 | ✅ 保留 | 匹配 Feature 定义 |
| 测试验证 | ✅ 过滤 | ✅ 过滤 | 不匹配任何类型 + 违反独立性 |
| 运维审批 | ✅ 过滤 | ✅ 过滤 | 不匹配任何类型 + 违反独立性 |

## 修改文件清单

| 文件 | 修改内容 |
|-----|---------|
| `graphiti_core/prompts/extract_nodes.py` | `filter_entities()` 增加 entity_types 上下文和决策流程 |
| `graphiti_core/utils/maintenance/node_operations.py` | `filter_extracted_nodes()` 新增 `entity_types_context` 参数 |
| `graphiti_core/utils/maintenance/node_operations.py` | `extract_nodes()` 调用时传入 `entity_types_context` |

---
