"""疾病 → 科室：统一走 disease_kb（OpenSearch 索引，本地 jsonl 兜底）。"""

from app.infra.opensearch_disease_kb import lookup_departments

__all__ = ["lookup_departments"]
