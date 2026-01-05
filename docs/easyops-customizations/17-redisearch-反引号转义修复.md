# 17. RediSearch 反引号转义修复

## 问题背景

导入包含代码片段的文档时，如果内容中包含反引号（Markdown 代码标记），在全文搜索时会导致 RediSearch 语法错误：

```
RediSearch: Syntax error at offset 115 near URL
```

错误的查询示例：
```
(user | info | API | URL | ` | ` | embed | dynamic | values)
```

## 修改内容

**文件**: `graphiti_core/helpers.py`

在 `lucene_sanitize()` 函数的 `escape_map` 中添加反引号转义：

```python
escape_map = str.maketrans(
    {
        # ... 其他字符
        '`': r'\`',  # EasyOps: escape backtick for RediSearch
        # ...
    }
)
```

## 修复效果

| 场景 | 修复前 | 修复后 |
|-----|-------|-------|
| 包含 Markdown 代码的文档 | RediSearch 语法错误 | 正常搜索 |

---
