#!/usr/bin/env python3
"""Merge recommendations into eval table CSV."""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPORTS = ROOT / "exports"
CSV_IN = EXPORTS / "medical_small_100_eval_table_v2.csv"
RECS = EXPORTS / "medical_small_100_recommendations.json"
CSV_OUT = EXPORTS / "medical_small_100_eval_table_v2.csv"


def main() -> int:
    recs = {r["idx"]: r for r in json.loads(RECS.read_text(encoding="utf-8"))}
    rows = list(csv.DictReader(CSV_IN.open(encoding="utf-8-sig")))
    extra = ["解决方案", "理想推荐科室", "推荐理由"]
    fieldnames = list(rows[0].keys()) + extra

    out_rows = []
    for row in rows:
        idx = int(row["#"])
        r = recs.get(idx, {})
        row["解决方案"] = r.get("fix_solution", "")
        row["理想推荐科室"] = r.get("ideal_dept", "")
        row["推荐理由"] = r.get("ideal_reason", "")
        out_rows.append(row)

    try:
        with CSV_OUT.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(out_rows)
        print(f"Updated {CSV_OUT} ({len(out_rows)} rows)")
    except PermissionError:
        alt = EXPORTS / "medical_small_100_eval_table_v3.csv"
        with alt.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(out_rows)
        print(f"Locked {CSV_OUT}, wrote {alt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
