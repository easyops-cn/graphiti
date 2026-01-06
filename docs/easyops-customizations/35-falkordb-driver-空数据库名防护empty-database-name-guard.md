# 35-FalkorDB Driver 空数据库名防护 (Empty Database Name Guard)

## 问题背景

FalkorDB 1.2.0 客户端的 `select_graph` 方法要求传入非空字符串作为数据库名称：

```python
def select_graph(self, graph_id: str):
    if not isinstance(graph_id, str) or graph_id == "":
        raise TypeError(f"Expected a string parameter, but received {type(graph_id)}.")
    return AsyncGraph(self, graph_id)
```

如果传入空字符串，会抛出 `TypeError` 错误，但错误信息有误导性（显示为 "received `<class 'str'>`"）。

在 FalkorDriver 的 `__init__` 方法中，会立即创建后台任务 `build_indices_and_constraints()`，该任务会调用 `execute_query()` → `_get_graph(self._database)`。如果 `self._database` 是空字符串，就会导致后台任务崩溃：

```
TypeError: Expected a string parameter, but received <class 'str'>.
```

## 修改内容

**文件**: `graphiti_core/driver/falkordb_driver.py`

在 `FalkorDriver.__init__` 方法中添加防御性检查：

```python
def __init__(
    self,
    host: str = 'localhost',
    port: int = 6379,
    username: str | None = None,
    password: str | None = None,
    falkor_db: FalkorDB | None = None,
    database: str = 'default_db',
    embedding_dimension: int = 1024,
):
    super().__init__()
    # Ensure database is not empty (FalkorDB 1.2.0 requires non-empty database name)
    if not database or not database.strip():
        logger.warning(f"Empty database name provided, using default 'default_db'")
        database = 'default_db'
    self._database = database
    # ... rest of init
```

## 影响范围

- 防止因空 `database` 参数导致的后台任务崩溃
- 自动回退到默认值 `'default_db'`，避免服务启动失败
- 添加警告日志，便于定位调用方问题

## 使用建议

调用方应确保传入有效的 `database` 参数（非空字符串），避免依赖这个兜底逻辑。

## 相关 Issue

- FalkorDB 1.2.0 的错误信息有歧义，建议在 upstream 提 issue 改进
