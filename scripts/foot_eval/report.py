from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = ROOT / "reports"


def write_report(payload: dict) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = REPORTS_DIR / f"foot_eval_{ts}.json"
    latest = REPORTS_DIR / "foot_eval_latest.json"
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    path.write_text(text, encoding="utf-8")
    latest.write_text(text, encoding="utf-8")

    # 明细单独一份，方便编辑器打开大文件
    details = payload.get("details")
    if details:
        detail_path = REPORTS_DIR / f"foot_eval_{ts}_details.json"
        detail_latest = REPORTS_DIR / "foot_eval_latest_details.json"
        detail_text = json.dumps(details, ensure_ascii=False, indent=2)
        detail_path.write_text(detail_text, encoding="utf-8")
        detail_latest.write_text(detail_text, encoding="utf-8")
        print(f"Details: {detail_latest}")

    print(f"\nReport: {latest}")
    return path


def print_summary(metrics: dict, overall: bool) -> None:
    print("\n" + "=" * 60)
    print("Foot Triage Eval Summary")
    print("=" * 60)
    for name, m in metrics.items():
        if not isinstance(m, dict):
            continue
        total = m.get("total", 0)
        if total == 0 and name != "live_smoke":
            print(f"{name}: (no cases)")
            continue
        status = "PASS" if m.get("ok", True) else "FAIL"
        print(
            f"[{status}] {name}: {m.get('passed', 0)}/{total} "
            f"rate={m.get('rate', 0):.1%} threshold={m.get('threshold', 'n/a')}"
        )
    if "live_smoke" in metrics:
        ls = metrics["live_smoke"]
        print(f"[INFO] live_smoke: {ls.get('passed', 0)}/{ls.get('total', 0)} (non-gating)")
    print("=" * 60)
    print(f"overall_ok: {overall}")
