"""
验证 spec §9 实体提取与路由（无 LLM 依赖的单元测试 + 可选 live）。

用法:
  python scripts/archive/test_entity_extract.py
  python scripts/archive/test_entity_extract.py --live
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.ner.extract import build_entity_result
from app.ner.models import EntityExtractResult, NERExtractOutput
from app.ner.span_utils import (
    is_valid_span,
    process_spans,
    resolve_overlapping_spans,
    select_primary_by_position,
)
from app.ner.triage_route import resolve_triage_route


def assert_substrings_in_query(result: EntityExtractResult) -> None:
    q = result.query
    for term in result.all_symptoms + result.all_diseases:
        assert term in q, f"{term!r} not in query {q!r}"


SPEC_CASES = [
    {
        "query": "最近心慌还手抖",
        "raw": NERExtractOutput(symptom_spans=["心慌", "手抖"], disease_spans=[]),
        "primary_symptom": "心慌",
        "companion_symptoms": ["手抖"],
        "primary_disease": None,
        "companion_diseases": [],
        "route": "symptom",
    },
    {
        "query": "我有胃炎",
        "raw": NERExtractOutput(symptom_spans=[], disease_spans=["胃炎"]),
        "primary_symptom": None,
        "companion_symptoms": [],
        "primary_disease": "胃炎",
        "companion_diseases": [],
        "route": "disease",
    },
    {
        "query": "胃炎还肚脐上方疼",
        "raw": NERExtractOutput(
            symptom_spans=["肚脐上方疼"], disease_spans=["胃炎"]
        ),
        "primary_symptom": "肚脐上方疼",
        "companion_symptoms": [],
        "primary_disease": "胃炎",
        "companion_diseases": [],
        "route": "disease",
    },
    {
        "query": "手抖心慌",
        "raw": NERExtractOutput(symptom_spans=["手抖", "心慌"], disease_spans=[]),
        "primary_symptom": "手抖",
        "companion_symptoms": ["心慌"],
        "primary_disease": None,
        "companion_diseases": [],
        "route": "symptom",
    },
    {
        "query": "你好",
        "raw": NERExtractOutput(symptom_spans=[], disease_spans=[]),
        "primary_symptom": None,
        "companion_symptoms": [],
        "primary_disease": None,
        "companion_diseases": [],
        "route": "reject",
    },
]


def test_span_utils() -> None:
    q = "肚脐上方疼痛"
    merged = resolve_overlapping_spans(["疼痛", "肚脐上方疼痛"], q)
    assert merged == ["肚脐上方疼痛"], merged

    primary, companions = select_primary_by_position(["手抖", "心慌"], "手抖心慌")
    assert primary == "手抖"
    assert companions == ["心慌"]

    assert not is_valid_span("心悸", "心里发慌")
    print("[OK] span_utils")


def test_spec_cases() -> None:
    for case in SPEC_CASES:
        result = build_entity_result(case["query"], case["raw"])
        assert result.primary_symptom == case["primary_symptom"], case
        assert result.companion_symptoms == case["companion_symptoms"], case
        assert result.primary_disease == case["primary_disease"], case
        assert result.companion_diseases == case["companion_diseases"], case
        assert resolve_triage_route(result) == case["route"], case
        assert_substrings_in_query(result)
    print("[OK] spec §9 cases")


def test_invalid_span_dropped() -> None:
    raw = NERExtractOutput(symptom_spans=["心悸", "心慌"], disease_spans=[])
    result = build_entity_result("最近心慌", raw)
    assert result.all_symptoms == ["心慌"]
    print("[OK] invalid span dropped")


def test_catalog_fallback() -> None:
    from app.ner.catalog_scan import load_entity_catalog, scan_catalog_substrings

    cat = load_entity_catalog()
    q = "我有胃炎还腹胀"
    raw = scan_catalog_substrings(q, cat["主症"], cat["疾病"])
    result = build_entity_result(q, raw)
    assert "胃炎" in result.all_diseases
    assert result.primary_disease == "胃炎"
    assert_substrings_in_query(result)
    print("[OK] catalog fallback")


def run_live() -> None:
    from app.ner.service import extract_entity_tags

    for case in SPEC_CASES:
        q = case["query"]
        if q == "你好":
            continue
        result = extract_entity_tags(q)
        payload = {
            "query": q,
            "primary_symptom": result.primary_symptom,
            "companion_symptoms": result.companion_symptoms,
            "primary_disease": result.primary_disease,
            "companion_diseases": result.companion_diseases,
            "route": resolve_triage_route(result),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        assert_substrings_in_query(result)
    print("[OK] live extract (substring invariant)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    test_span_utils()
    test_spec_cases()
    test_invalid_span_dropped()
    test_catalog_fallback()
    if args.live:
        run_live()
    else:
        print("Skip live tests (use --live)")


if __name__ == "__main__":
    main()
