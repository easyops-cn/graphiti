# 4. Small Model 支持

## 问题背景

Graphiti 的 `OpenAIClient` 支持 `small_model` 参数，可以为简单任务使用较小的模型以节省成本。但 Elevo Memory 使用的 `OpenAIGenericClient`（用于非官方 OpenAI 兼容端点）没有实现这个功能，导致 `model_size` 参数被忽略。

## 修改内容

### 4.1 添加 _get_model_for_size 方法

**文件**: `graphiti_core/llm_client/openai_generic_client.py`

```python
def _get_model_for_size(self, model_size: ModelSize) -> str:
    """Get the appropriate model name based on the requested size."""
    if model_size == ModelSize.small:
        return self.small_model or self.model or DEFAULT_MODEL
    else:
        return self.model or DEFAULT_MODEL
```

### 4.2 修改 _generate_response 使用该方法

```python
# 原来
response = await self.client.chat.completions.create(
    model=self.model or DEFAULT_MODEL,
    ...
)

# 修改后
model = self._get_model_for_size(model_size)
response = await self.client.chat.completions.create(
    model=model,
    ...
)
```

## 配置方法

在 `.env` 文件或环境变量中配置：

```bash
# 主模型（复杂任务）
OPENAI_MODEL=qwen3-235a22b-2507-local_2

# 小模型（简单任务，可选）
OPENAI_SMALL_MODEL=qwen3-235a22b-2507-local_2
```

如果不设置 `OPENAI_SMALL_MODEL`，系统会自动使用 `OPENAI_MODEL` 作为 fallback。

---
