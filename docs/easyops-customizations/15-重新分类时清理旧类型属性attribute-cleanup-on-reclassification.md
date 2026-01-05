# 15. 重新分类时清理旧类型属性（Attribute Cleanup on Reclassification）

## 问题背景

当实体从一个类型重新分类到另一个类型时，原有代码只更新了 `node.labels` 和 `node.reasoning`，但没有清理 `node.attributes` 中属于旧类型的属性。

**具体案例**：

```
1. 实体 "流程库" 初始被错误分类为 ProductModule
2. 属性抽取阶段提取了 ProductModule 属性：module_name="EasyITSC", code="ITSC"
3. 两步验证检测到误分类，重新分类为 Feature
4. node.labels 更新为 ['Entity', 'Feature']
5. 但 node.attributes 仍然包含 module_name, code（属于 ProductModule）
6. 这些污染属性被写入数据库并通过去重传播
```

**最终结果**：数据库中 Feature 实体包含 ProductModule 的属性（属性污染）。

## 解决方案

**文件**: `graphiti_core/graphiti.py`

**位置**: `add_episode()` 方法和 `_resolve_nodes_and_edges_bulk()` 方法中的重分类逻辑

在更新 `node.labels` 后，根据新类型的 Schema 清理不属于新类型的属性：

```python
# Apply reclassifications to nodes
if entities_to_reclassify:
    reclassify_map = {r.name: (r.new_type, r.reason) for r in entities_to_reclassify}
    for node in hydrated_nodes:
        if node.name in reclassify_map:
            new_type, reason = reclassify_map[node.name]
            old_labels = node.labels.copy()
            node.labels = ['Entity', new_type] if new_type != 'Entity' else ['Entity']

            # Clear attributes not belonging to new type schema
            # This prevents attribute pollution when reclassifying from one type to another
            if entity_types and new_type in entity_types:
                new_type_model = entity_types[new_type]
                valid_fields = set(new_type_model.model_fields.keys())
                old_attrs = node.attributes.copy()
                node.attributes = {k: v for k, v in node.attributes.items() if k in valid_fields}
                if old_attrs != node.attributes:
                    removed_attrs = set(old_attrs.keys()) - set(node.attributes.keys())
                    logger.info(f'Cleared invalid attributes from "{node.name}": {removed_attrs}')
            elif new_type == 'Entity':
                # Reclassified to generic Entity, clear all custom attributes
                if node.attributes:
                    logger.info(f'Cleared all attributes from "{node.name}" (reclassified to Entity)')
                    node.attributes = {}

            # Update reasoning with reclassification reason
            node.reasoning = f'[Reclassified from {old_labels} to {node.labels}] {reason}'
```

## 清理逻辑

| 新类型 | 清理策略 |
|-------|---------|
| 有 Schema 定义的类型（如 Feature） | 只保留新类型 Schema 中定义的字段 |
| 通用 Entity 类型 | 清除所有自定义属性 |

## 日志输出

- `Cleared invalid attributes from "流程库": {'module_name', 'code'}`
- `Cleared all attributes from "xxx" (reclassified to Entity)`

## 效果

| 场景 | 修复前 | 修复后 |
|-----|-------|-------|
| ProductModule → Feature | attributes 保留 module_name, code | 只保留 feature_name, description 等 |
| Feature → Entity | attributes 保留 feature_name | attributes 清空 |
| Component → Feature | attributes 保留 component_type | 只保留 Feature Schema 字段 |

## 修改文件清单

| 文件 | 修改内容 |
|-----|---------|
| `graphiti_core/graphiti.py` | `add_episode()` 重分类后清理属性 |
| `graphiti_core/graphiti.py` | `_resolve_nodes_and_edges_bulk()` 批量重分类后清理属性 |

---
