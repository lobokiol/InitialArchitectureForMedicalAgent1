"""Repair mojibake in tests/results_batch100.json without re-running eval."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = ROOT / "tests" / "cases" / "batch_100_cases.json"
RESULTS_PATH = ROOT / "tests" / "results_batch100.json"


def load_cases_by_id() -> dict[str, dict]:
    data = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    return {c["id"]: c for c in data["cases"]}


def decode_mojibake(text: str) -> str:
    """UTF-8 bytes were mis-read as GBK, then saved back as UTF-8."""
    try:
        return text.encode("gbk").decode("utf-8")
    except UnicodeError:
        # Some corrupted private-use chars cannot round-trip; keep best-effort fix.
        return text.encode("gbk", errors="ignore").decode("utf-8", errors="ignore")


def fix_value(value):
    if isinstance(value, str):
        return decode_mojibake(value)
    if isinstance(value, list):
        return [fix_value(v) for v in value]
    if isinstance(value, dict):
        return {fix_value(k): fix_value(v) for k, v in value.items()}
    return value


def fix_json_text(raw: str) -> str:
    """Restore missing closing quotes and remove trailing commas."""
    raw = re.sub(r'"([^"\r\n]*?)\?,(\s*\r?\n)', r'"\1",\2', raw)
    raw = re.sub(r'"([^"\r\n]*?)\?(\s*\r?\n)', r'"\1",\2', raw)
    raw = re.sub(r",(\s*\])", r"\1", raw)
    raw = re.sub(r",(\s*})", r"\1", raw)
    return raw


def main() -> int:
    cases_by_id = load_cases_by_id()
    raw_bytes = RESULTS_PATH.read_bytes()
    if raw_bytes.startswith(b"\xef\xbb\xbf"):
        raw_bytes = raw_bytes[3:]

    raw = raw_bytes.decode("utf-8")
    raw = fix_json_text(raw)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        debug_path = RESULTS_PATH.with_suffix(".repair_attempt.json")
        debug_path.write_text(raw, encoding="utf-8")
        raise SystemExit(f"JSON still invalid after repair: {exc} (see {debug_path.name})") from exc

    data = fix_value(data)

    for item in data.get("results", []):
        case_id = item.get("id")
        if case_id in cases_by_id:
            item["query"] = cases_by_id[case_id]["query"]

    RESULTS_PATH.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Repaired {RESULTS_PATH} ({len(data.get('results', []))} results)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
