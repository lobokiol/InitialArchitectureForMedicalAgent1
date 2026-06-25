"""
独立 NER 提取测试脚本（不依赖主 LangGraph / Redis / OpenSearch）。

用法:
  python scripts/archive/test_ner_extract.py
  python scripts/archive/test_ner_extract.py "自定义句子"
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.ner.service import extract_entity_tags

DEFAULT_QUERY = (
    "最近每次饭后，肚脐上面那一块儿一阵一阵绞着疼，还腹胀，有胃炎"
)


def main() -> None:
    query = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_QUERY
    try:
        result = extract_entity_tags(query)
        error = None
    except Exception as exc:
        result = None
        error = str(exc)

    payload = {
        "query": query,
        "主要症状": result.primary_symptom if result else None,
        "伴随症状": result.companion_symptoms if result else [],
        "主要疾病": result.primary_disease if result else None,
        "伴随疾病": result.companion_diseases if result else [],
        "error": error,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
