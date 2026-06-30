#!/usr/bin/env python3
"""Summarize scope-A eval results from _eval_input.json."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "exports" / "_eval_input.json"

TARGET = {6, 7, 8, 10, 22, 26, 48, 81, 84, 87, 97, 98}
WANT = {
    6: "妇科",
    7: "妇科",
    8: "妇科",
    10: "儿科",
    22: "妇科",
    26: "妇科",
    48: "儿科",
    81: "妇科",
    84: "妇科",
    87: "儿科",
    97: "妇科",
    98: "儿科",
}


def main() -> int:
    if not EVAL.exists():
        print(f"Missing {EVAL}; run build_medical_eval_table.py first")
        return 1
    cases = json.loads(EVAL.read_text(encoding="utf-8"))
    sym_unmatched = [c for c in cases if c.get("route") == "symptom" and c.get("outcome") == "unmatched"]
    print(f"symptom+unmatched total: {len(sym_unmatched)}")
    print("--- target 12 ---")
    ok = 0
    for c in cases:
        idx = c["idx"]
        if idx not in TARGET:
            continue
        dept = c.get("dept")
        outcome = c.get("outcome")
        want = WANT[idx]
        good = dept == want and outcome == "locked"
        ok += int(good)
        mark = "OK" if good else "FAIL"
        print(
            f"#{idx} {mark} outcome={outcome} dept={dept} want={want} "
            f"chunk={c.get('chunk_id')}"
        )
    print(f"--- target locked correct dept: {ok}/12 ---")
    print("--- other symptom+unmatched ---")
    for c in sym_unmatched:
        if c["idx"] not in TARGET:
            q = (c.get("query") or "")[:55]
            print(f"  #{c['idx']} {q} dept={c.get('dept')} chunk={c.get('chunk_id')}")
    return 0 if ok == 12 else 1


if __name__ == "__main__":
    raise SystemExit(main())
