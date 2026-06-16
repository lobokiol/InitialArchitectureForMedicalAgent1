"""
Golden 用例准确度评估：实体提取(NER) + 三分类意图路由。

用法:
  python tests/run_eval.py
  python tests/run_eval.py --cases tests/cases/golden_cases.json
  python tests/run_eval.py --id G04_symptom_palpitation
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from langchain_core.messages import HumanMessage

from app.domain.models import AppState
from app.graph.nodes.decision import decision_node
from app.ner.models import NERExtractResult
from app.ner.service import extract_entity_tags
from app.ner.triage_route import resolve_triage_route

DEFAULT_CASES = Path(__file__).resolve().parent / "cases" / "golden_cases.json"
GOLDEN_CASES = Path(__file__).resolve().parent / "cases" / "golden_cases.json"
BATCH_100_CASES = Path(__file__).resolve().parent / "cases" / "batch_100_cases.json"

# 标注与抽取结果的常见同义/近义
_TERM_ALIASES: dict[str, set[str]] = {
    "头疼": {"头痛", "头疼"},
    "头痛": {"头疼", "头痛"},
    "发烧": {"发热", "发烧", "高热"},
    "发热": {"发烧", "发热", "高热"},
    "畏寒": {"怕冷", "畏寒", "寒战"},
    "怕冷": {"畏寒", "怕冷", "寒战"},
    "心悸": {
        "心悸", "心慌", "心里发慌", "心跳厉害", "心里难受", "心累", "落空感", "心跳快",
        "心跳加速", "心跳不齐", "心律不齐", "心扑通扑通跳", "心咚咚跳", "心突突跳",
        "心乱", "心焦", "心跳声大", "能听见心跳", "早搏感", "漏跳感",
    },
    "心慌": {
        "心悸", "心慌", "心里发慌", "心跳厉害", "心里难受", "心累", "落空感", "心跳快",
        "心跳加速", "心跳不齐", "心律不齐", "心扑通扑通跳", "心咚咚跳", "心突突跳",
        "心乱", "心焦", "心跳声大", "能听见心跳", "早搏感", "漏跳感",
    },
    "心跳快": {
        "心悸", "心慌", "心里发慌", "心跳厉害", "心里难受", "心累", "落空感", "心跳快",
        "心跳加速", "心跳不齐", "心律不齐", "心扑通扑通跳", "心咚咚跳", "心突突跳",
        "心乱", "心焦", "心跳声大", "能听见心跳", "早搏感", "漏跳感",
    },
    "胃疼": {"腹痛", "胃疼", "上腹痛", "饭后上腹痛"},
    "腹痛": {"腹痛", "胃疼", "肚子疼", "脐上疼痛", "右下腹疼痛"},
    "腹泻": {"腹泻", "拉肚子"},
    "拉肚子": {"腹泻", "拉肚子"},
    "乏力": {"乏力", "疲劳", "没劲", "虚弱", "易疲劳"},
    "疲劳": {"乏力", "疲劳", "没劲"},
    "没劲": {"乏力", "疲劳", "没劲"},
    "流鼻涕": {"流鼻涕", "流涕", "流清鼻涕"},
    "鼻塞": {"鼻塞", "鼻子堵"},
    "嗓子疼": {"嗓子疼", "咽痛", "吞咽疼痛"},
    "胸闷": {"胸闷", "胸口闷", "胸口不适"},
    "气短": {"气短", "喘不上气", "气喘", "喘息"},
    "气喘": {"气短", "气喘", "喘息", "喘不上气"},
    "瘙痒": {"瘙痒", "痒", "很痒"},
    "红疹": {"红疹", "皮疹", "起包", "皮肤起包"},
    "甲亢": {"甲亢", "甲状腺功能亢进"},
    "甲状腺功能亢进": {"甲亢", "甲状腺功能亢进"},
}


@dataclass
class CaseResult:
    case_id: str
    query: str
    passed: bool
    intent_ok: bool
    ner_ok: bool
    failures: list[str] = field(default_factory=list)
    actual: dict = field(default_factory=dict)
    entity_recall: float | None = None


def _expand_term(term: str) -> set[str]:
    t = term.strip()
    if not t:
        return set()
    aliases = set(_TERM_ALIASES.get(t, {t}))
    aliases.add(t)
    return aliases


def _term_matches(expected: str, actual: str) -> bool:
    e_set = _expand_term(expected)
    a_set = _expand_term(actual)
    if e_set & a_set:
        return True
    for e in e_set:
        for a in a_set:
            if e in a or a in e:
                return True
    return False


def _entity_recall(expected: list[str], actual: list[str]) -> tuple[float, list[str]]:
    if not expected:
        return 1.0, []
    missed = [e for e in expected if not any(_term_matches(e, a) for a in actual)]
    hit = len(expected) - len(missed)
    return hit / len(expected), missed


def _collect_symptom_terms(ner: NERExtractResult) -> list[str]:
    terms: list[str] = []
    for field_name in ("symptom_candidates", "companion_symptoms"):
        terms.extend(getattr(ner, field_name) or [])
    if ner.chief_symptom:
        terms.append(ner.chief_symptom)
    if ner.chief_symptom_canonical:
        terms.append(ner.chief_symptom_canonical)
    return terms


def _contains_all(actual: list[str], expected: list[str]) -> bool:
    actual_set = {x.strip() for x in actual if x and str(x).strip()}
    return all(item in actual_set for item in expected)


def _diseases_match(actual: list[str], expected: list[str]) -> bool:
    if not expected:
        return not actual
    actual_norm = {x.strip() for x in actual}
    expected_norm = {x.strip() for x in expected}
    return expected_norm.issubset(actual_norm)


def _check_ner(expect: dict, ner: NERExtractResult) -> list[str]:
    failures: list[str] = []

    if "diseases" in expect and not _diseases_match(ner.diseases, expect["diseases"]):
        failures.append(f"diseases: expected {expect['diseases']}, got {ner.diseases}")

    if "chief_symptom" in expect and ner.chief_symptom != expect["chief_symptom"]:
        failures.append(
            f"chief_symptom: expected {expect['chief_symptom']!r}, got {ner.chief_symptom!r}"
        )

    if "chief_symptom_any_of" in expect:
        allowed = expect["chief_symptom_any_of"]
        if ner.chief_symptom not in allowed:
            failures.append(
                f"chief_symptom: expected one of {allowed}, got {ner.chief_symptom!r}"
            )

    if "chief_symptom_canonical" in expect:
        if ner.chief_symptom_canonical != expect["chief_symptom_canonical"]:
            failures.append(
                "chief_symptom_canonical: "
                f"expected {expect['chief_symptom_canonical']!r}, "
                f"got {ner.chief_symptom_canonical!r}"
            )

    if "slot_table_code" in expect and ner.slot_table_code != expect["slot_table_code"]:
        failures.append(
            f"slot_table_code: expected {expect['slot_table_code']!r}, got {ner.slot_table_code!r}"
        )

    if "symptom_candidates_contains" in expect:
        if not _contains_all(ner.symptom_candidates, expect["symptom_candidates_contains"]):
            failures.append(
                "symptom_candidates: "
                f"expected to contain {expect['symptom_candidates_contains']}, "
                f"got {ner.symptom_candidates}"
            )

    if expect.get("symptom_candidates_empty") and ner.symptom_candidates:
        failures.append(
            f"symptom_candidates: expected empty, got {ner.symptom_candidates}"
        )

    if "symptom_entities" in expect:
        recall, missed = _entity_recall(expect["symptom_entities"], _collect_symptom_terms(ner))
        min_recall = expect.get("symptom_entity_min_recall", 0.5)
        if recall < min_recall:
            failures.append(
                f"symptom_entities recall {recall:.0%} < {min_recall:.0%}, "
                f"missed {missed}, got {_collect_symptom_terms(ner)}"
            )

    if "disease_entities" in expect:
        recall, missed = _entity_recall(expect["disease_entities"], ner.diseases)
        min_recall = expect.get("disease_entity_min_recall", 1.0)
        if recall < min_recall:
            failures.append(
                f"disease_entities recall {recall:.0%} < {min_recall:.0%}, "
                f"missed {missed}, got {ner.diseases}"
            )

    return failures


def evaluate_case(case: dict) -> CaseResult:
    case_id = case["id"]
    query = case.get("query", "")
    expect = case.get("expect", {})
    failures: list[str] = []

    if not query.strip():
        route = "reject"
        ner = NERExtractResult(query=query)
    else:
        ner = extract_entity_tags(query)
        route = resolve_triage_route(ner)

    actual = {
        "triage_route": route,
        "diseases": ner.diseases,
        "chief_symptom": ner.chief_symptom,
        "chief_symptom_canonical": ner.chief_symptom_canonical,
        "slot_table_code": ner.slot_table_code,
        "symptom_candidates": ner.symptom_candidates,
        "companion_symptoms": ner.companion_symptoms,
        "hints": ner.hints,
    }

    intent_ok = True
    if "triage_route" in expect and route != expect["triage_route"]:
        intent_ok = False
        failures.append(f"triage_route: expected {expect['triage_route']!r}, got {route!r}")

    ner_failures = _check_ner(expect, ner)
    ner_ok = not ner_failures
    failures.extend(ner_failures)

    entity_recall = None
    recalls: list[float] = []
    if "symptom_entities" in expect:
        r, _ = _entity_recall(expect["symptom_entities"], _collect_symptom_terms(ner))
        recalls.append(r)
    if "disease_entities" in expect:
        r, _ = _entity_recall(expect["disease_entities"], ner.diseases)
        recalls.append(r)
    if recalls:
        entity_recall = sum(recalls) / len(recalls)

    return CaseResult(
        case_id=case_id,
        query=query,
        passed=intent_ok and ner_ok,
        intent_ok=intent_ok,
        ner_ok=ner_ok,
        failures=failures,
        actual=actual,
        entity_recall=entity_recall,
    )


def evaluate_decision_node(case: dict) -> bool:
    """decision_node 端到端：intent_result.triage_route 与 NER 路由一致。"""
    query = case.get("query", "")
    out = decision_node(AppState(messages=[HumanMessage(content=query)]))
    ir = out.get("intent_result")
    route = ir.triage_route if ir else None
    expect_route = case.get("expect", {}).get("triage_route")
    return route == expect_route


def load_cases(path: Path, case_id: str | None = None) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    cases = data.get("cases", [])
    if case_id:
        cases = [c for c in cases if c["id"] == case_id]
        if not cases:
            raise SystemExit(f"Case not found: {case_id}")
    return cases


def _golden_queries() -> set[str]:
    try:
        return {c["query"].strip() for c in load_cases(GOLDEN_CASES)}
    except Exception:
        return set()


def filter_duplicate_cases(cases: list[dict], skip_queries: set[str]) -> tuple[list[dict], list[str]]:
    kept: list[dict] = []
    skipped: list[str] = []
    for case in cases:
        q = case.get("query", "").strip()
        if q in skip_queries:
            skipped.append(case["id"])
        else:
            kept.append(case)
    return kept, skipped


def print_report(
    results: list[CaseResult],
    decision_checks: list[tuple[str, bool]],
    *,
    brief: bool = False,
    skipped: list[str] | None = None,
) -> None:
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    intent_pass = sum(1 for r in results if r.intent_ok)
    ner_pass = sum(1 for r in results if r.ner_ok)
    decision_pass = sum(1 for _, ok in decision_checks if ok)

    print("\n" + "=" * 60)
    print("准确度报告 / Accuracy Report")
    print("=" * 60)
    recalls = [r.entity_recall for r in results if r.entity_recall is not None]
    avg_entity_recall = sum(recalls) / len(recalls) if recalls else None

    print(f"用例总数:     {total}")
    if skipped:
        print(f"跳过重复:     {len(skipped)} ({', '.join(skipped)})")
    print(f"意图路由准确: {intent_pass}/{total} ({100 * intent_pass / total:.1f}%)")
    print(f"实体提取准确: {ner_pass}/{total} ({100 * ner_pass / total:.1f}%)")
    if avg_entity_recall is not None:
        print(f"实体平均召回: {100 * avg_entity_recall:.1f}%")
    print(f"综合通过:     {passed}/{total} ({100 * passed / total:.1f}%)")
    if decision_checks:
        d_total = len(decision_checks)
        print(
            f"decision_node: {decision_pass}/{d_total} "
            f"({100 * decision_pass / d_total:.1f}%)"
        )
    print("=" * 60)

    show = [r for r in results if not brief or not r.passed]
    for r in show:
        status = "PASS" if r.passed else "FAIL"
        print(f"\n[{status}] {r.case_id}")
        print(f"  query: {r.query!r}")
        if r.entity_recall is not None:
            print(f"  entity_recall: {r.entity_recall:.0%}")
        if r.failures:
            for msg in r.failures:
                print(f"  ✗ {msg}")
        if not brief:
            print(f"  actual: {json.dumps(r.actual, ensure_ascii=False)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="NER + 意图路由 Golden 评估")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--id", dest="case_id", default=None, help="只跑单个用例")
    parser.add_argument("--json", action="store_true", help="输出 JSON 结果")
    parser.add_argument("--skip-decision-node", action="store_true", help="跳过 decision_node 二次调用")
    parser.add_argument("--brief", action="store_true", help="仅输出失败用例详情")
    parser.add_argument(
        "--skip-duplicates",
        action="store_true",
        help="跳过与 golden_cases.json 中 query 完全重复的用例",
    )
    args = parser.parse_args()

    cases = load_cases(args.cases, args.case_id)
    skipped: list[str] = []
    if args.skip_duplicates:
        cases, skipped = filter_duplicate_cases(cases, _golden_queries())
    results: list[CaseResult] = []
    decision_checks: list[tuple[str, bool]] = []

    for case in cases:
        result = evaluate_case(case)
        results.append(result)
        if not args.skip_decision_node:
            ok = evaluate_decision_node(case)
            decision_checks.append((case["id"], ok))
            if not ok and result.intent_ok:
                result.failures.append("decision_node triage_route mismatch")
                result.passed = False

    if args.json:
        payload = {
            "summary": {
                "total": len(results),
                "passed": sum(1 for r in results if r.passed),
                "intent_accuracy": sum(1 for r in results if r.intent_ok) / len(results),
                "ner_accuracy": sum(1 for r in results if r.ner_ok) / len(results),
            },
            "results": [
                {
                    "id": r.case_id,
                    "query": r.query,
                    "passed": r.passed,
                    "intent_ok": r.intent_ok,
                    "ner_ok": r.ner_ok,
                    "failures": r.failures,
                    "actual": r.actual,
                }
                for r in results
            ],
        }
        if skipped:
            payload["skipped"] = skipped
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print_report(results, decision_checks, brief=args.brief, skipped=skipped or None)

    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
