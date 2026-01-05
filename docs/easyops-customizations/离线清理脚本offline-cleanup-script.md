# 离线清理脚本（Offline Cleanup Script）

## 问题背景

两轮 Map-Reduce 语义去重只能处理：
1. 新节点 vs 新节点
2. 新节点 vs 数据库候选

但无法处理：数据库中已存在的重复节点（历史数据）。

如果数据库中已经存在多个重复的 ProductModule 节点（如 `EasyITSM`、`IT服务中心`、`ITSC服务中心`），这些节点不会被自动合并。

## 解决方案

提供离线清理脚本 `scripts/cleanup_duplicate_nodes.py`，使用 LLM 识别并合并数据库中的重复实体。

## 使用方法

```bash
# 干运行（不做实际修改，只显示会做什么）
python scripts/cleanup_duplicate_nodes.py --group-id easyops_support --dry-run

# 清理特定实体类型
python scripts/cleanup_duplicate_nodes.py --group-id easyops_support --entity-type ProductModule

# 清理所有实体类型
python scripts/cleanup_duplicate_nodes.py --group-id easyops_support

# 指定数据库连接
python scripts/cleanup_duplicate_nodes.py \
    --group-id easyops_support \
    --falkordb-host localhost \
    --falkordb-port 6379 \
    --falkordb-database elevo_memory
```

## 工作流程

1. **查询节点**：从数据库查询指定 group_id 的所有实体节点，按类型分组
2. **识别重复**：对每种类型的节点，使用 LLM 识别语义重复的节点对
3. **合并节点**：
   - 将源节点的所有边重定向到规范节点
   - 合并源节点的 summary 和 attributes 到规范节点
   - 删除源节点

## 环境变量

| 变量 | 说明 |
|-----|------|
| `OPENAI_API_KEY` | LLM API Key |
| `OPENAI_MODEL` | LLM 模型名称 |
| `OPENAI_BASE_URL` | LLM API 端点（可选） |
| `FALKORDB_HOST` | FalkorDB 主机 |
| `FALKORDB_PORT` | FalkorDB 端口 |
| `FALKORDB_DATABASE` | 数据库名称 |

## 注意事项

1. **先干运行**：建议先用 `--dry-run` 查看会合并哪些节点
2. **备份数据**：执行前建议备份数据库
3. **LLM 成本**：每对节点比较都需要调用 LLM，大量节点会产生较高成本
4. **执行时间**：O(n²) 复杂度，节点数多时耗时较长

## 修改文件清单

| 文件 | 修改内容 |
|-----|---------|
| `scripts/cleanup_duplicate_nodes.py` | 新增离线清理脚本 |

---
