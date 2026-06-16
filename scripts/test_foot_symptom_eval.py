"""
Deprecated wrapper — use scripts/run_foot_triage_eval.py

  uv run python scripts/run_foot_triage_eval.py --offline
  uv run python scripts/run_foot_triage_eval.py --offline --live
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_foot_triage_eval import main

if __name__ == "__main__":
    if "--offline" not in sys.argv and "--graph" not in sys.argv and "--all" not in sys.argv:
        sys.argv.insert(1, "--offline")
        if "--live" in sys.argv:
            pass
    raise SystemExit(main())
