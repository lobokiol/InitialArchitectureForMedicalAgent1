# OpenSearch rag_knowledge 语义召回设计

## 目标

在 OpenSearch 中对 `demo/data/rag_knowledge.jsonl`（12 条脚部症状）实现语义召回，验收：

| Query | 期望 top1 |
|-------|-----------|
| 脚心出汗 | RK0010 |
| 脚出汗 | RK0010 |

范围仅限召回片段，不含追问、科室推理、拒答策略。

## 约束

- 使用 OpenSearch kNN / hybrid
- 不使用 Elasticsearch、Milvus
- 向量模型：DashScope `text-embedding-v2`（1536 维）

## 入库

1. 读取 jsonl 每行原始 JSON 字符串 `raw_line`
2. `doc = json.loads(raw_line)` 写入 `_source`（完整字段）
3. `embedding = embed(raw_line)` — 向量输入为**原始 JSON 行**
4. `_id` = `doc["id"]`（RK0001–RK0012）

## 查询

默认 `recall(query, k=3)` 使用 **hybrid**：

- **BM25**：`alliance^4`、`canonical_symptom^5`、`description^2`
- **kNN**：`embedding` 字段，query 向量由用户输入 embed
- **融合**：`rag-knowledge-hybrid-pipeline`（min_max 归一化 + arithmetic_mean）

## 返回

```json
{
  "query": "脚出汗",
  "hits": [
    {"id": "RK0010", "score": 0.87, "canonical_symptom": "足部多汗 / 足底多汗", "body_part": "脚"}
  ]
}
```

## 文件

| 文件 | 职责 |
|------|------|
| `demo/opensearch_mappings.py` | 索引 mapping |
| `demo/opensearch_rag_kb.py` | 入库、recall、hybrid pipeline |
| `esTools/test_rag_recall.py` | 验收脚本 |
