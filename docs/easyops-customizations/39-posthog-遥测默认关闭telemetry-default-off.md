# PostHog 遥测默认关闭（true → false）

## 问题背景

Graphiti 上游默认开启 PostHog 遥测（`GRAPHITI_TELEMETRY_ENABLED` 默认 `true`），
每次图操作后向 `https://us.i.posthog.com` 上报事件。

企业内网环境（如百丽）不可达外网 PostHog：
- 每次 `capture` 触发的上报在后台 consumer 线程排队/重试
- cProfile 采样显示 bulk 导入中 `posthog.consumer.upload` 占比显著
  （117s 的 profile 里 posthog upload 累计 ~110s，虽多数在后台线程，
  flush 阻塞与重试放大了整体延迟）

## 修改内容

`graphiti_core/telemetry/telemetry.py`:

```python
# 修改前
env_value = os.environ.get(TELEMETRY_ENV_VAR, 'true').lower()
# 修改后
env_value = os.environ.get(TELEMETRY_ENV_VAR, 'false').lower()
```

默认关闭；需要开启时显式设置 `GRAPHITI_TELEMETRY_ENABLED=true`。

## 验证

基准测试（3874 实体 + 3000 边 upsert 重放）中已确认 posthog 不再产生
upload 活动；行为由环境变量显式控制。

## 升级注意事项

上游 Graphiti 新版本若重写 telemetry 模块，合并时保留此默认值修改；
或改为部署侧统一注入 `GRAPHITI_TELEMETRY_ENABLED=false` 环境变量（
docker-compose / k8s env），后者对上游升级免疫。
