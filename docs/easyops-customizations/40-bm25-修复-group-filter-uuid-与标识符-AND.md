# BM25 修复：group_id UUID 过滤失效 + 标识符查询 OR 噪声

## 问题背景

两个叠加 bug 导致 BM25 全文检索长期失效/低质（用户反馈"搜 IP 返回一堆不相关 db"的根因链）：

### Bug 1：group_id 过滤恒 0 命中（BM25 整路静默失效）

`build_fulltext_query` 把租户过滤拼进 fulltext 串：`(@group_id:439b6035-cd73-...)`。
FalkorDB 分词器按"空白+标点"切分，**UUID 的连字符被当分隔符**，`439b6035-cd73-...`
被拆碎，精确匹配永远为空 → BM25 对 UUID 型 group_id 的所有查询返回 0 条。
表现为：搜索结果全靠 embedding 单路支撑，质量下降且无告警。

### Bug 2：多词 OR 展开对标识符查询是噪声海洋

上游 (#914) 把多词查询拼成 OR（`10 | 252 | 12 | 40`）。IP 查询被 sanitize 拆成
数字段后，OR 语义下单含 `10` 即命中（实测 6786 条），真目标被淹没。

## 修改内容

`graphiti_core/driver/falkordb_driver.py` `build_fulltext_query`：

1. **group_id 不再拼进 fulltext 串**。租户过滤由已有的 Cypher 后置
   `WHERE n.group_id IN $group_ids` 承担（search_utils 两处调用点原本就有，
   之前是 fulltext 串里坏过滤 + Cypher 好过滤并存，坏的那份导致 0 命中短路）。
   注：FalkorDB 的 queryNodes 不带 limit，全量 YIELD 后过滤再 LIMIT，语义不变；
   OR 自然语言查询的 YIELD 量会增大（实测同量级图上毫秒级完成，无感）。
2. **标识符特征判定 `_is_identifier_query`**：任一 token 含 IP 片段形态
   （`\d+\.\d+`）或 ≥2 个纯数字 token → 空格 join（AND）；其余维持上游 OR。
   AND 要求全部 token 命中，实测 IP 查询正确目标排到第 1 位。

## 验证

- 单测 `tests/unit/test_bm25_fix.py`（7 例：group filter 移除、IP/多数字 AND、
  自然语言/单词保持 OR）
- 生产实测（修复前 → 后）：见任务记录。IP "10.252.12.40" 从全噪声 db 结果
  → 命中 `s:host:lida-12-40`（summary 富化 + 本修复组合生效）

## 升级注意事项

合并上游新版 falkordb_driver.py 时保留：
- build_fulltext_query 的 group filter 移除（若上游重写，确认调用方 Cypher
  后置过滤仍在）
- _is_identifier_query 及 joiner 分支

上游相关（未解决，勿等）：
- FalkorDB#314 tokenization（挂 3 年零回应）
- RediSearch#1084 tokenizer 插件化（挂 6 年）
- FalkorDB PR#2527 TAG 索引类型（新 Rust 引擎，未合并）——合并后可重估方案
