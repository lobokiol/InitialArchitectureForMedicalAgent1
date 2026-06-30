#!/usr/bin/env python3
"""Run eval rounds 3..7 until metric A >= 80% or max round reached."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_a(round_num: int) -> float:
    p = ROOT / "exports" / f"medical_small_100_eval_summary_v{round_num}.json"
    if not p.exists():
        return 0.0
    return json.loads(p.read_text(encoding="utf-8")).get("metric_A_recommend_rate", 0.0)


def main() -> int:
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    end = int(sys.argv[2]) if len(sys.argv) > 2 else 7
    for r in range(start, end + 1):
        print(f"\n########## ROUND {r} ##########")
        rc = subprocess.call([sys.executable, "scripts/run_eval_round.py", "--round", str(r)], cwd=ROOT)
        a = read_a(r)
        print(f"Round {r} metric_A={a:.2%}")
        if a >= 0.80:
            print(f"Target A>=80% met at round {r}")
            return 0
        if rc != 0 and a < 0.80:
            print(f"Round {r} finished with issues (exit {rc})")
    return 1 if read_a(end) < 0.80 else 0


if __name__ == "__main__":
    raise SystemExit(main())
