# 8. 属性抽取枚举验证容错（Attribute Extraction Graceful Handling）

## 问题背景

LLM 抽取实体属性时，可能返回不在 Schema 枚举值中的字段值，导致 Pydantic 验证失败，整个批次导入崩溃。

**具体案例**：

```
ValidationError: 1 validation error for Component
component_type
  Input should be 'backend', 'middleware', 'database', 'frontend', 'script', 'storage' or 'agent'
  [type=literal_error, input_value='feature', input_type=str]
```

LLM 把 `component_type` 填成了 `'feature'`，但 Schema 只允许特定的枚举值。

## 解决方案

**文件**: `graphiti_core/utils/maintenance/node_operations.py`

修改 `_extract_entity_attributes()` 函数，添加 `ValidationError` 的容错处理：

1. 捕获 `ValidationError` 而不是直接抛出
2. 从错误信息中提取验证失败的字段名
3. 从响应中移除这些无效字段
4. 返回清理后的响应

```python
from pydantic import BaseModel, ValidationError

async def _extract_entity_attributes(...) -> dict[str, Any]:
    # ... 省略 ...

    # validate response with graceful error handling for invalid enum values
    try:
        entity_type(**llm_response)
    except ValidationError as e:
        # EasyOps customization: handle invalid enum values gracefully
        logger.warning(f'Entity attribute validation warning: {e}. Will remove invalid fields.')

        # Extract field names that have validation errors
        invalid_fields = set()
        for error in e.errors():
            if error.get('loc'):
                field_name = error['loc'][0]
                invalid_fields.add(field_name)
                logger.warning(f'Removing invalid field "{field_name}"')

        # Remove invalid fields and try again
        cleaned_response = {k: v for k, v in llm_response.items() if k not in invalid_fields}

        # Validate cleaned response
        try:
            entity_type(**cleaned_response)
            return cleaned_response
        except ValidationError as e2:
            logger.error(f'Validation still failed after cleanup: {e2}')
            return {}  # Return empty dict rather than crash

    return llm_response
```

## 效果

- 当 LLM 返回无效枚举值时，只丢弃该字段，保留其他有效属性
- 批量导入不会因为单个字段验证失败而崩溃
- 日志记录被移除的字段，便于后续优化 Schema 或 Prompt

## 修改文件清单

| 文件 | 修改内容 |
|-----|---------|
| `graphiti_core/utils/maintenance/node_operations.py` | 添加 `ValidationError` import |
| `graphiti_core/utils/maintenance/node_operations.py` | `_extract_entity_attributes()` 添加枚举验证容错逻辑 |
| `graphiti_core/prompts/extract_nodes.py` | `filter_entities()` 增加 Schema 实体类型上下文 |
| `graphiti_core/utils/maintenance/node_operations.py` | `filter_extracted_nodes()` 传入 entity_types_context |

---
