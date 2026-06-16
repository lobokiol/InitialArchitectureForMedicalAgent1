"""
独立 NER 提取测试脚本（不依赖主 LangGraph / Redis / ES / Milvus）。

用法:
  python scripts/test_ner_extract.py
  python scripts/test_ner_extract.py "自定义句子"
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from langgraph.graph import END, START, StateGraph

from app.graph.nodes.ner_extract import ner_extract_node
from app.ner.models import NERExtractState

DEFAULT_QUERY = (
    "最近每次饭后，肚脐上面那一块儿一阵一阵绞着疼，还腹胀，有胃炎"
)


def build_ner_test_graph():
    graph = StateGraph(NERExtractState)
    graph.add_node("ner_extract", ner_extract_node)
    graph.add_edge(START, "ner_extract")
    graph.add_edge("ner_extract", END)
    return graph.compile()


def main() -> None:
    query = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_QUERY
    app = build_ner_test_graph()
    out = app.invoke({"query": query})

    payload = {
        "query": query,
        "主要症状": out.get("primary_symptom"),
        "伴随症状": out.get("companion_symptoms", []),
        "主要疾病": out.get("primary_disease"),
        "伴随疾病": out.get("companion_diseases", []),
        "error": out.get("error"),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
