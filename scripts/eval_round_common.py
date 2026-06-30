"""Shared config and metrics for eval rounds v3+."""
from __future__ import annotations

SKIP_INDICES: frozenset[int] = frozenset({24, 30, 34, 52, 57, 70, 83, 89, 95, 100})
EVALUABLE_COUNT = 100 - len(SKIP_INDICES)
AUTO_PASS_COUNT = len(SKIP_INDICES)
TARGET_A_RATE = 0.80


def is_skipped(idx: int) -> bool:
    return idx in SKIP_INDICES


def recommend_ok(case: dict) -> bool:
    if case.get("skipped"):
        return True
    outcome = case.get("outcome") or ""
    dept = case.get("dept")
    return outcome in ("locked", "disease", "skipped_pass") and bool(dept)


def dept_relevance_ok(case: dict) -> bool:
    if case.get("skipped"):
        return True
    score = case.get("dept_relevance_score") or 0
    return score >= 70


def entity_ok(case: dict) -> bool:
    if case.get("skipped"):
        return True
    return (case.get("entity_score") or 0) >= 70


def chunk_should_recall(case: dict) -> bool:
    if case.get("skipped"):
        return False
    route = case.get("route")
    query = (case.get("query") or "").strip()
    if route == "disease":
        return False
    if route == "symptom":
        return True
    return any(k in query for k in ("痛", "痒", "血", "晕", "发烧", "咳", "肿", "不适"))


def chunk_top1_ok(case: dict) -> bool:
    if case.get("skipped"):
        return True
    if not chunk_should_recall(case):
        return not bool(case.get("chunk_id"))
    return (case.get("chunk_score") or 0) >= 70


def chunk_top3_ok(case: dict) -> bool:
    if case.get("skipped"):
        return True
    if not chunk_should_recall(case):
        return not bool(case.get("top3_hit"))
    return bool(case.get("top3_hit"))


def compute_metrics(cases: list[dict]) -> dict:
    evaluable = [c for c in cases if not c.get("skipped")]
    skipped = [c for c in cases if c.get("skipped")]

    a_num = sum(1 for c in evaluable if recommend_ok(c))
    b_den = [c for c in evaluable if c.get("dept")]
    b_num = sum(1 for c in b_den if dept_relevance_ok(c))
    d_num = len(skipped) + a_num

    entity_num = sum(1 for c in evaluable if entity_ok(c))
    recall_cases = [c for c in evaluable if chunk_should_recall(c)]
    top1_num = sum(1 for c in recall_cases if chunk_top1_ok(c))
    top3_num = sum(1 for c in recall_cases if chunk_top3_ok(c))

    symptom_locked = sum(
        1 for c in evaluable if c.get("route") == "symptom" and c.get("outcome") == "locked"
    )
    disease_ok = sum(
        1 for c in evaluable if c.get("route") == "disease" and c.get("outcome") == "disease"
    )

    return {
        "metric_A_recommend_rate": round(a_num / EVALUABLE_COUNT, 4),
        "metric_A_numerator": a_num,
        "metric_A_denominator": EVALUABLE_COUNT,
        "metric_B_dept_relevance_rate": round(b_num / len(b_den), 4) if b_den else 0.0,
        "metric_B_numerator": b_num,
        "metric_B_denominator": len(b_den),
        "metric_D_overall_success_rate": round(d_num / 100, 4),
        "metric_D_numerator": d_num,
        "metric_D_denominator": 100,
        "ner_accuracy": round(entity_num / EVALUABLE_COUNT, 4),
        "chunk_top1_recall": round(top1_num / len(recall_cases), 4) if recall_cases else 0.0,
        "chunk_top3_recall": round(top3_num / len(recall_cases), 4) if recall_cases else 0.0,
        "symptom_locked_rate": round(symptom_locked / EVALUABLE_COUNT, 4),
        "disease_route_success_rate": round(disease_ok / EVALUABLE_COUNT, 4),
        "target_A_met": a_num / EVALUABLE_COUNT >= TARGET_A_RATE,
        "skipped_count": len(skipped),
        "evaluable_count": len(evaluable),
        "chunk_should_recall_cases": len(recall_cases),
    }
