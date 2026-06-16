from __future__ import annotations

THRESHOLDS = {
    "A_recall": 0.90,
    "B_dept": 0.85,
    "C_disease": 0.90,
    "E_emergency": 1.0,
    "D_false_push_max": 0.05,
    "D_reject": 0.90,
    "F_multiturn": 0.85,
}


def rate(passed: int, total: int) -> float:
    return passed / total if total else 1.0


def metric_entry(name: str, passed: int, total: int, threshold: float, *, higher_is_better: bool = True) -> dict:
    r = rate(passed, total)
    if total == 0:
        ok = True
    elif higher_is_better:
        ok = r >= threshold
    else:
        ok = r <= threshold
    return {
        "passed": passed,
        "total": total,
        "rate": round(r, 4),
        "threshold": threshold,
        "ok": ok,
    }


def build_d_false_push_metrics(d_results: list[dict]) -> dict:
    if not d_results:
        return metric_entry("D_false_push", 0, 0, THRESHOLDS["D_false_push_max"], higher_is_better=False)
    false_push = sum(1 for r in d_results if r.get("locked_department"))
    total = len(d_results)
    r = false_push / total
    ok = r <= THRESHOLDS["D_false_push_max"]
    return {
        "passed": total - false_push,
        "total": total,
        "rate": round(r, 4),
        "threshold": THRESHOLDS["D_false_push_max"],
        "ok": ok,
        "false_push_count": false_push,
    }


def overall_ok(metrics: dict) -> bool:
    for key, m in metrics.items():
        if key == "live_smoke":
            continue
        if isinstance(m, dict) and m.get("total", 0) > 0 and not m.get("ok", True):
            return False
    return True
