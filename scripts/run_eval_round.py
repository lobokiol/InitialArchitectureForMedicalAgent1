#!/usr/bin/env python3
"""Orchestrate one eval round: reindex -> batch(90) -> report v{N}."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval_round_common import SKIP_INDICES

SKIP_STR = ",".join(str(i) for i in sorted(SKIP_INDICES))

ROUND_CHANGELOG = {
    3: "R1: 收紧 CL0017，移除独立症状别名，改用复合儿科短语",
    4: "R2: 收紧 CL0016；新增 CL0018 精神心理 + RK0180",
    5: "R3: 新增 CL0019 男性泌尿 + CL0020 咳痰出血 + RK0190/RK0200",
    6: "R4: 新增 CL0021 皮肤疱疹；CL0005 腰痛强化；RK0201",
    7: "R5: dept_rules 消化/口腔/心内 differential 精调",
}


def run(cmd: list[str], desc: str) -> None:
    print(f"\n=== {desc} ===")
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--round", type=int, required=True, help="Report version 3-7")
    p.add_argument("--skip-batch", action="store_true")
    p.add_argument("--skip-reindex", action="store_true")
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = p.parse_args()
    ver = args.round
    if ver < 3 or ver > 7:
        print("round must be 3..7")
        return 2
    user_id = f"batch-med-small-100-r{ver}"

    if not args.skip_reindex:
        run([sys.executable, "sourceData/opensearch_rag_kb.py", "--skip-acceptance"], "reindex rag_knowledge")
        run([sys.executable, "sourceData/opensearch_dept_rules.py"], "reindex dept_rules")

    if not args.skip_batch:
        rc = subprocess.call(
            [
                sys.executable,
                "scripts/run_medical_data_batch.py",
                "--user-id",
                user_id,
                "--skip-indices",
                SKIP_STR,
                "--base-url",
                args.base_url,
            ],
            cwd=ROOT,
        )
        if rc != 0:
            print(f"[warn] batch finished with errors (exit {rc}), continuing to report")

    changelog = ROUND_CHANGELOG.get(ver, "")
    rc = subprocess.call(
        [
            sys.executable,
            "scripts/build_eval_round_report.py",
            "--version",
            str(ver),
            "--user-id",
            user_id,
            "--changelog",
            changelog,
        ],
        cwd=ROOT,
    )
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
