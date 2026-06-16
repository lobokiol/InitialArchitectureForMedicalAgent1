"""
脚部导诊 Golden Set 统一评测。

用法:
  uv run python scripts/run_foot_triage_eval.py --offline
  uv run python scripts/run_foot_triage_eval.py --graph --subset B,C,E
  uv run python scripts/run_foot_triage_eval.py --reindex --all
  uv run python scripts/run_foot_triage_eval.py --offline --graph --live
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.foot_eval.graph_runner import run_graph_eval
from scripts.foot_eval.live import run_live_smoke
from scripts.foot_eval.loader import build_golden_from_legacy, load_cases
from scripts.foot_eval.metrics import (
    THRESHOLDS,
    build_d_false_push_metrics,
    metric_entry,
    overall_ok,
)
from scripts.foot_eval.offline import run_offline_recall
from scripts.foot_eval.report import print_summary, write_report
from scripts.foot_eval.version import check_opensearch_indices, collect_versions, reindex_all


def _parse_subsets(raw: str | None) -> set[str] | None:
    if not raw:
        return None
    return {s.strip().upper() for s in raw.split(",") if s.strip()}


def main() -> int:
    parser = argparse.ArgumentParser(description="Foot triage Golden Set eval")
    parser.add_argument("--offline", action="store_true", help="subset A recall")
    parser.add_argument("--graph", action="store_true", help="subset B/C/D/E/F graph")
    parser.add_argument("--live", action="store_true", help="optional /chat smoke (non-gating)")
    parser.add_argument("--all", action="store_true", help="offline + graph")
    parser.add_argument("--reindex", action="store_true", help="reindex rag + disease kb")
    parser.add_argument("--build-golden", action="store_true", help="regenerate golden jsonl")
    parser.add_argument("--subset", default=None, help="comma subsets e.g. A,B,E")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--skip-embed-check", action="store_true")
    parser.add_argument(
        "--brief",
        action="store_true",
        help="omit per-case details from JSON report (summary only)",
    )
    args = parser.parse_args()

    if args.build_golden:
        build_golden_from_legacy()
        print("Built demo/data/foot_triage_golden.jsonl")

    if args.reindex:
        reindex_all()

    run_offline = args.offline or args.all
    run_graph = args.graph or args.all
    if not (run_offline or run_graph or args.live):
        run_offline = True

    subsets = _parse_subsets(args.subset)
    cases = load_cases(subsets)
    versions = collect_versions()
    opensearch_info: dict = {}
    if args.reindex or run_offline or run_graph:
        opensearch_info = check_opensearch_indices(skip_embed_check=args.skip_embed_check)
        if opensearch_info.get("errors"):
            for err in opensearch_info["errors"]:
                print(f"[ERROR] {err}")
            if not args.skip_embed_check:
                return 1

    metrics: dict = {}
    failures: list[dict] = []
    detail: dict = {}

    if run_offline:
        a_cases = [c for c in cases if c.get("subset") == "A"]
        results, passed, total = run_offline_recall(a_cases)
        metrics["A_recall"] = metric_entry("A_recall", passed, total, THRESHOLDS["A_recall"])
        detail["A_recall"] = results
        failures.extend(r for r in results if not r["ok"])

    if run_graph:
        graph_subsets = {"B", "C", "D", "E", "F"}
        g_cases = [c for c in cases if c.get("subset") in graph_subsets]
        if g_cases:
            results, passed, total = run_graph_eval(g_cases)
            detail["graph"] = results
            failures.extend(r for r in results if not r["ok"])

            by_sub: dict[str, list] = {}
            for r in results:
                by_sub.setdefault(r["subset"], []).append(r)

            if "B" in by_sub:
                rs = by_sub["B"]
                p = sum(1 for r in rs if r["ok"])
                metrics["B_dept"] = metric_entry("B_dept", p, len(rs), THRESHOLDS["B_dept"])
            if "C" in by_sub:
                rs = by_sub["C"]
                p = sum(1 for r in rs if r["ok"])
                metrics["C_disease"] = metric_entry("C_disease", p, len(rs), THRESHOLDS["C_disease"])
            if "D" in by_sub:
                rs = by_sub["D"]
                p = sum(1 for r in rs if r["ok"])
                metrics["D_reject"] = metric_entry("D_reject", p, len(rs), THRESHOLDS["D_reject"])
                metrics["D_false_push"] = build_d_false_push_metrics(rs)
            if "E" in by_sub:
                rs = by_sub["E"]
                p = sum(1 for r in rs if r["ok"])
                metrics["E_emergency"] = metric_entry("E_emergency", p, len(rs), THRESHOLDS["E_emergency"])
            if "F" in by_sub:
                rs = by_sub["F"]
                p = sum(1 for r in rs if r["ok"])
                metrics["F_multiturn"] = metric_entry("F_multiturn", p, len(rs), THRESHOLDS["F_multiturn"])

    if args.live:
        live_cases = [c for c in cases if c.get("subset") in {"A", "B", "C", "E"}]
        try:
            l_results, l_passed, l_total = run_live_smoke(
                live_cases, base_url=args.base_url, timeout=args.timeout
            )
            metrics["live_smoke"] = {
                "passed": l_passed,
                "total": l_total,
                "rate": round(l_passed / l_total, 4) if l_total else 0,
                "ok": True,
            }
            detail["live"] = l_results
        except Exception as exc:
            print(f"[WARN] live smoke skipped: {exc}")
            metrics["live_smoke"] = {"passed": 0, "total": 0, "ok": True, "error": str(exc)}

    ok = overall_ok(metrics)
    print_summary(metrics, ok)

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "versions": versions,
        "opensearch": opensearch_info,
        "metrics": metrics,
        "failures": failures,
        "overall_ok": ok,
    }
    if not args.brief and detail:
        payload["details"] = detail
    write_report(payload)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
