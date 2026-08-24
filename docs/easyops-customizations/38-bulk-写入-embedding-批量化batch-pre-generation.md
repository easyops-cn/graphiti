# Bulk 写入链路 embedding 批量化（batch pre-generation）

## 问题背景

Bulk / Topology 批量写入时，每条实体（name embedding）和每条边（fact embedding）
都**逐条调用** `graphiti.embedder.create()`：

- `_create_entity_node`: `entity.py:662` name embedding 逐条生成
- `_create_entity_edge`: `entity.py:1265/1306` fact embedding 逐条生成

实测（one.elevo.vip 网关，2026-08-19）：单次 embeddings 请求 ~100ms，串行逐条导致：

- 单批 100 条边 ≈ 10s，其中 ~8s 在等 embedding HTTP 往返（占 80%）
- 全量 36196 条边需 ~75 分钟，其中 ~60 分钟纯等 embedding

而 OpenAI 兼容 embeddings API 原生支持 `input` 数组（一次请求多个文本），
graphiti 的 `OpenAIEmbedder.create_batch()` 也已实现，只是 Bulk 链路没用上。

## 修改内容

`src/elevo_memory/services/entity.py`：

1. `add_entities_bulk()` 在批处理循环前新增 **4.5 步：批量预生成**：
   - 实体：本批全部 `name` 一次 `create_batch()` → `dict[original_idx -> vector]`
   - 边：本批全部 fact（`"{source_name} {edge_type} {target_name}"`）一次
     `create_batch()` → `dict[original_idx -> vector]`
   - 预生成失败（网络抖动等）时**回退逐条生成**，写入不中断
2. `_create_entity_node()` / `_create_entity_edge()` 新增 `precomputed_embedding`
   参数：传入时直接使用，未传（None）时保持原逐条行为（向后兼容 Triplet 等
   单条调用路径）
3. fact 文本拼接逻辑与原实现完全一致，向量结果等价（同一 API，同一文本，
   仅请求聚合方式不同）

## 效果

单批 embedding HTTP 调用从 N 次（N=实体数+边数，≤100）降为 1 次。
预计批量导入吞吐提升 5-8 倍（36196 条边 ~75 分钟 → ~10-15 分钟，
剩余耗时为端点查找与图写入，非 embedding）。

## 验证

- 单元测试：`tests/unit/test_bulk_embedding_batch.py`
  - 实体批处理聚合为一次 create_batch，且不触发逐条 create
  - 边 fact embedding 聚合为一次 create_batch
- 回归：entity/ingest/bulk_dedup/reranker 相关 115 个单测全过

## 升级注意事项

该改动在 elevo_memory 主项目（非 graphiti 子模块）。若上游 graphiti 的
`OpenAIEmbedder.create_batch` 行为变化（如返回顺序不再与输入一致），
`zip(strict=True)` 会显式报错而非静默错位，届时按报错处理。
