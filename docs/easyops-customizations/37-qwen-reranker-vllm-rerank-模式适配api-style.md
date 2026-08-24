# Qwen Reranker 支持 vLLM rerank 模式部署（api_style 适配）

## 问题背景

`QwenRerankerClient` 原实现只支持**生成式部署**的 Qwen3-Reranker：通过
`/v1/chat/completions` + `logprobs` 提取 "yes" token 概率作为相关性分数。

客户环境（belle llm-gateway）的 `qwen3-reranker-0.6b` 是按 **vLLM rerank 模式**
（`--task rerank`）部署的：

- `/v1/chat/completions` 请求返回 HTTP 200 但 body 为 `{"object": "error", ...}`，
  无论是否带 logprobs 参数都一样
- `/v1/rerank` 端点正常工作，返回标准 `results[].relevance_score`

即原客户端在此类网关上全部拿到中性分 0.5，reranking 完全失效。

## 修改内容

`graphiti_core/cross_encoder/qwen_reranker_client.py`：

1. `QwenRerankerConfig` 新增两个字段：
   - `api_style: str = 'chat'` — `'chat'`（原行为，默认）或 `'rerank'`（vLLM rerank 模式）
   - `batch_size: int = 100` — rerank 风格下单次请求的文档数上限
2. `rank()` 按 `api_style` 分流：
   - `chat`：原逻辑不变（并发逐文档 `_score_single`）
   - `rerank'`：新增 `_rank_via_rerank_endpoint()`，按 `batch_size` 分批调用
     `POST {base_url}/v1/rerank`，payload 为 `{"model", "query", "documents": [...]}`，
     按 `results[].index` 还原输入顺序，单批失败该批全部回退中性分 0.5
3. 错误兜底语义与原实现一致：异常时返回 0.5，不中断搜索流程

## 配置方法

主项目环境变量（`src/elevo_memory/core/config.py` 透传）：

```bash
QWEN_RERANKER_ENABLED=true
QWEN_RERANKER_BASE_URL=http://llm-gateway.prd.bjm6v.belle.lan   # 注意不带 /v1
QWEN_RERANKER_MODEL=qwen3-reranker-0.6b
QWEN_RERANKER_API_KEY=sk-xxx
QWEN_RERANKER_API_STYLE=rerank   # chat（默认）= 生成式部署；rerank = vLLM rerank 模式
QWEN_RERANKER_BATCH_SIZE=100     # 可选，默认 100
```

## 验证

- 单元测试：`tests/unit/test_qwen_reranker_client.py`（chat 与 rerank 两种风格的
  URL/payload/排序/错误兜底/分批行为）
- 真实网关实测：belle llm-gateway 上 `qwen3-embedding-0.6b`（/v1/embeddings）与
  `qwen3-reranker-0.6b`（/v1/rerank）均返回正常结果（2026-08-19）

## 升级注意事项

升级 Graphiti 上游版本时，若上游也提供了 rerank 端点适配，可对比该文件的
`api_style` 分流逻辑，保留配置字段名以兼容现有部署。
