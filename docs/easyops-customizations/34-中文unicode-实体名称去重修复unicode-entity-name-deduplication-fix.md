# 34. 中文/Unicode 实体名称去重修复（Unicode Entity Name Deduplication Fix）

## 问题背景

重复导入相同的文本数据时，中文实体名称（如 "张三"、"电商事业部"）会创建重复节点，而英文实体名称能正确去重。

**复现场景**：
```python
# 第一轮导入
episodes = ["张三是订单服务的负责人，他在电商事业部工作。", ...]
add_episode(episodes)
# 结果：创建 张三、电商事业部 等实体

# 第二轮导入（相同内容）
add_episode(episodes)
# 期望：识别为重复，不创建新节点
# 实际：又创建了 张三、电商事业部 等重复节点！
```

## 根因分析

**文件**: `graphiti_core/utils/maintenance/dedup_helpers.py`

1. **`_normalize_name_for_fuzzy()` 只支持 ASCII**

   ```python
   # 原代码（第 47 行）
   normalized = re.sub(r"[^a-z0-9' ]", ' ', _normalize_string_exact(name))
   ```

   正则 `[^a-z0-9' ]` 只保留小写字母、数字、撇号和空格，**所有中文字符被删除**：
   - `"order-service-prod"` → `"order service prod"` ✅
   - `"张三"` → `""` ❌ 变成空字符串

2. **空字符串无法通过熵值检查**

   `_has_high_entropy("")` 返回 False，因为：
   - 长度 = 0（小于 `_MIN_NAME_LENGTH = 6`）
   - token 数 = 0（小于 `_MIN_TOKEN_COUNT = 2`）

3. **熵值检查失败导致跳过精确匹配**

   原代码先检查熵值，再尝试精确匹配：
   ```python
   # 原代码逻辑（简化）
   if not _has_high_entropy(normalized_fuzzy):
       state.unresolved_indices.append(idx)  # 跳过！
       continue

   # 精确匹配代码永远不会被执行
   existing_matches = indexes.normalized_existing.get(normalized_exact, [])
   ```

   对于中文名称，熵值检查失败后直接跳过，精确匹配代码**根本不执行**，导致完全相同的中文名称无法被去重。

## 解决方案

### 34.1 修复 `_normalize_name_for_fuzzy()` 支持 Unicode

使用 `\w` 匹配 Unicode 字母数字（Python 3 默认支持 Unicode）：

```python
def _normalize_name_for_fuzzy(name: str) -> str:
    """Produce a fuzzier form that keeps alphanumerics and apostrophes for n-gram shingles.

    EasyOps fix: Support Unicode characters (Chinese, Japanese, Korean, etc.) by using
    Unicode-aware regex. The original [a-z0-9] only matched ASCII letters and digits,
    which stripped all CJK characters and caused deduplication failures for non-ASCII names.
    """
    normalized = _normalize_string_exact(name)
    # Use \w to match Unicode letters/digits (includes underscore)
    # [^\w' ]|_ means: match non-word chars (except apostrophe/space) OR underscore
    # This keeps Unicode letters, digits, apostrophes, and spaces
    normalized = re.sub(r"[^\w' ]|_", ' ', normalized, flags=re.UNICODE)
    normalized = normalized.strip()
    return re.sub(r'[\s]+', ' ', normalized)
```

**正则解释**：
- `\w` 匹配所有 Unicode 字母、数字、下划线
- `[^\w' ]` 匹配非词字符（除了撇号和空格）
- `[^\w' ]|_` 匹配非词字符或下划线（即只保留字母、数字、撇号、空格）

### 34.2 修复 `_resolve_with_similarity()` 先精确匹配

即使修复了 `_normalize_name_for_fuzzy`，短名称（如 "张三" 只有 2 个字符）仍会因长度不足无法通过熵值检查。因此需要**先尝试精确匹配**：

```python
def _resolve_with_similarity(extracted_nodes, indexes, state):
    """Attempt deterministic resolution using exact name hits and fuzzy MinHash comparisons.

    EasyOps fix: Check exact match BEFORE entropy check to support non-ASCII names (e.g., Chinese).
    """
    for idx, node in enumerate(extracted_nodes):
        normalized_exact = _normalize_string_exact(node.name)
        normalized_fuzzy = _normalize_name_for_fuzzy(node.name)

        # EasyOps fix: Always try exact match first, regardless of entropy.
        # This ensures non-ASCII names (Chinese, Japanese, etc.) are deduplicated correctly.
        existing_matches = indexes.normalized_existing.get(normalized_exact, [])
        if len(existing_matches) == 1:
            match = existing_matches[0]
            state.resolved_nodes[idx] = match
            state.uuid_map[node.uuid] = match.uuid
            if match.uuid != node.uuid:
                state.duplicate_pairs.append((node, match))
            continue
        if len(existing_matches) > 1:
            # Multiple exact matches - let LLM decide
            state.unresolved_indices.append(idx)
            continue

        # No exact match found. Check entropy before fuzzy matching.
        if not _has_high_entropy(normalized_fuzzy):
            state.unresolved_indices.append(idx)
            continue

        # ... fuzzy matching logic ...
```

## 效果

| 实体名称 | 修复前 | 修复后 |
|---------|-------|-------|
| 张三 | 重复创建 | ✅ 精确匹配去重 |
| 李四 | 重复创建 | ✅ 精确匹配去重 |
| 电商事业部 | 重复创建 | ✅ 精确匹配去重 |
| 订单服务 | 重复创建 | ✅ 精确匹配去重 |
| order-service-prod | ✅ 正常去重 | ✅ 正常去重 |

## 验证方法

```bash
# 运行诊断脚本
poetry run python scripts/diagnose_dedup.py

# 运行复现测试
poetry run python scripts/reproduce_episode_simple.py
# 预期输出：✅ 未发现重复实体
```

## 修改文件清单

| 文件 | 修改内容 |
|-----|---------|
| `graphiti_core/utils/maintenance/dedup_helpers.py` | `_normalize_name_for_fuzzy()` 使用 Unicode 感知正则 |
| `graphiti_core/utils/maintenance/dedup_helpers.py` | `_resolve_with_similarity()` 先精确匹配后检查熵值 |

## 升级注意事项

1. **代码修改后需重启服务**：修改 Python 代码后，需重启 uvicorn 服务使修改生效
2. **历史重复数据**：此修复只影响新导入的数据，已存在的重复数据需使用清理脚本处理：
   ```bash
   poetry run python scripts/cleanup_duplicate_nodes.py --group-id <group_id>
   ```