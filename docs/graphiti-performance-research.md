# Graphiti 性能基准与数据规模调研报告

> 调研日期: 2025-12-10

## 结论摘要

**Graphiti 目前没有公开的大规模数据量 benchmark 报告**。官方测试数据集和性能声明都集中在相对中等规模的会话数据上。

---

## 1. 官方测试的数据规模

### 1.1 LongMemEval 数据集（代码库中存在）

**位置**: `tests/evals/data/longmemeval_data/longmemeval_oracle.json`

| 指标 | 数值 |
|-----|------|
| 问题数量 | 500 |
| 会话数量 | 948 |
| 消息总数 | 10,960 |
| 文件大小 | 15MB |
| 平均上下文长度 | ~115,000 tokens |

### 1.2 DMR (Deep Memory Retrieval) 基准

| 指标 | 数值 |
|-----|------|
| 对话数量 | 500 |
| 每个对话会话数 | 5 |
| 每会话消息数 | 最多 12 条 |
| 总消息数 | ~30,000 |

**关键发现**: 两个官方测试都是围绕**数百个多轮会话**进行的，并未测试百万级节点/边的图规模。

---

## 2. 已公开的性能指标

### 2.1 检索延迟

| 来源 | 延迟指标 |
|-----|---------|
| Zep 商业版 | P95 延迟 **300ms** |
| Graphiti 开源版 | 通常 **< 1 秒** (sub-second) |
| GraphRAG 对比 | "数秒到数十秒" |

### 2.2 准确率 (DMR 基准)

| 系统 | 模型 | 准确率 |
|-----|------|--------|
| Zep (Graphiti) | gpt-4o-mini | **98.2%** |
| Zep (Graphiti) | gpt-4-turbo | **94.8%** |
| MemGPT | gpt-4-turbo | 93.4% |

### 2.3 LongMemEval 基准

| 配置 | 准确率 | 延迟 | 上下文 tokens |
|-----|--------|------|--------------|
| Zep + gpt-4o | **71.2%** | 2.58s | 1.6k |
| Zep + gpt-4o-mini | 63.8% | 3.20s | 1.6k |
| Full-context baseline | 60.2% | 28.9s | 115k |

---

## 3. 代码库中的负载测试配置

**位置**: `mcp_server/tests/test_stress_load.py`

### 默认负载测试参数

```python
LoadTestConfig:
    num_clients: 10          # 并发客户端数
    operations_per_client: 100  # 每客户端操作数
    ramp_up_time: 5.0s       # 预热时间
    test_duration: 60.0s     # 测试持续时间
```

### 测试断言阈值

| 指标 | 阈值 |
|-----|------|
| 平均延迟 | < 5.0s |
| P95 延迟 | < 10.0s |
| 最小吞吐量 | > 1.0 ops/s |
| 高负载成功率 | > 50% |
| 内存泄漏警告 | 增长 > 100MB |

### 负载测试场景

1. **test_sustained_load**: 5 客户端，30 秒持续负载
2. **test_spike_load**: 50 并发突发操作
3. **test_memory_leak_detection**: 100 操作，10 批次
4. **test_connection_pool_exhaustion**: 100 并发长连接
5. **test_gradual_degradation**: 5→10→20→40→80 递增并发

---

## 4. 已知的性能瓶颈

### 4.1 build_communities OOM 问题

**GitHub Issue #992**: `build_communities` 函数对大图执行 N 次独立查询，会导致内存溢出崩溃。

### 4.2 previous_episodes 内容重复

**位置**: `graphiti_core/utils/maintenance/node_operations.py`, `edge_operations.py`

重复的 episode 内容增加 token 消耗和 LLM 处理时间。

---

## 5. 扩展性架构特点

| 特性 | 说明 |
|-----|------|
| 并发控制 | SEMAPHORE_LIMIT (默认 20，MCP 默认 10) |
| 混合检索 | 语义向量 + BM25 + 图遍历，避免 LLM 调用 |
| 索引优化 | 向量和 BM25 索引提供"近乎常数时间"访问 |
| 后端支持 | Neo4j 5.26+, FalkorDB 1.1.2+, Kuzu, Neptune |

---

## 6. 结论与建议

### 6.1 官方测试规模

- **最大测试**: ~10,000 消息 / ~1,000 会话 / ~115K tokens 上下文
- **未公开**: 百万级节点/边的图规模测试结果

### 6.2 性能声明

- **P95 延迟**: 300ms (商业版) / sub-second (开源版)
- **扩展性声明**: "可处理大规模知识图谱"，但无具体数据支撑

### 6.3 建议

如需在大规模场景使用 Graphiti，建议：
1. 自行进行规模测试 (>100K 节点)
2. 关注 `build_communities` 的 OOM 问题
3. 根据 LLM 配额调整 SEMAPHORE_LIMIT

---

## 参考来源

- [arXiv 论文: Zep: A Temporal Knowledge Graph Architecture](https://arxiv.org/abs/2501.13956)
- [Neo4j Blog: Graphiti Knowledge Graph Memory](https://neo4j.com/blog/developer/graphiti-knowledge-graph-memory/)
- [GitHub Issue #992: OOM in build_communities](https://github.com/getzep/graphiti/issues/992)
- 代码库: `tests/evals/`, `mcp_server/tests/test_stress_load.py`
