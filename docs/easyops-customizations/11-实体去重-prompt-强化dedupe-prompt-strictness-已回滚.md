# 11. 实体去重 Prompt 强化（Dedupe Prompt Strictness）- **已回滚**

## 问题背景

原有去重 prompt 导致 LLM 将相关但不同的实体错误合并：
- 不同操作被合并（如 "Approve" vs "Reject"）
- 同类别不同功能被合并（如 "User Management" vs "Role Management"）

## 解决方案（已回滚）

**文件**: `graphiti_core/prompts/dedupe_nodes.py`

曾在 `nodes()` 函数中增加严格的判断标准：

```python
# STRICT DUPLICATE CRITERIA

Entities are duplicates ONLY if they refer to **the EXACT same real-world object or concept**.

**TRUE DUPLICATES** (should merge):
- Same entity, different names (e.g., abbreviation vs full name, or translation)
- Typos or minor spelling variations of the same entity
- Synonyms that refer to the identical concept in this domain

**NOT DUPLICATES** (keep separate):
- Different operations/actions in the same domain
- Different features in the same category
- Related but conceptually distinct items
- Parent-child or hierarchical relationships
- Items with similar names but different purposes or scopes

# DECISION RULE

When uncertain, **DO NOT merge**. It is better to have two separate entities
than to incorrectly merge distinct concepts.

Ask yourself: "Are these two names referring to the IDENTICAL thing,
or are they two different things that happen to be related?"
```

## 回滚原因（2025-12-12）

上述修改太过保守，导致 LLM 不合并明显应该合并的实体：
- EasyITSM 和 EasyITSC 不被合并，尽管它们的 `module_name="EasyITSC"` 和 `code="ITSC"` 完全一致
- LLM 看到名字不同就"uncertain"，然后因为 "When uncertain, DO NOT merge" 而不合并

**已恢复到 Graphiti 原始 prompt**：

```python
Entities should only be considered duplicates if they refer to the *same real-world object or concept*.

Do NOT mark entities as duplicates if:
- They are related but distinct.
- They have similar names or purposes but refer to separate instances or concepts.
```

原始 prompt 更简洁，没有"When uncertain, DO NOT merge"这种极端保守的规则，LLM 可以正确利用 attributes（如 module_name、code）来判断重复。

## 修改文件清单

| 文件 | 修改内容 |
|-----|---------|
| `graphiti_core/prompts/dedupe_nodes.py` | ~~`nodes()` 增加 STRICT DUPLICATE CRITERIA 和 DECISION RULE~~ **已回滚** |

---
