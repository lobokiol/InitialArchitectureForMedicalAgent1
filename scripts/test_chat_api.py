"""
对已运行的 FastAPI 服务做导诊功能测试。

用法:
  python scripts/test_chat_api.py
  python scripts/test_chat_api.py --base-url http://localhost:8000
"""
from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.domain.triage_intent import REJECT_MESSAGE

CASES = [
    {
        "name": "纯疾病",
        "message": "我有胃炎",
        "expect_route": "disease",
    },
    {
        "name": "纯症状",
        "message": "最近有点心慌手抖",
        "expect_route": "symptom",
    },
    {
        "name": "病+症",
        "message": "胃炎还肚脐上方疼",
        "expect_route": "disease",
    },
    {
        "name": "拒答",
        "message": "你好",
        "expect_route": "reject",
        "expect_reply": REJECT_MESSAGE,
    },
    {
        "name": "非医疗",
        "message": "怎么预约",
        "expect_route": "reject",
        "expect_reply": REJECT_MESSAGE,
    },
    {
        "name": "踝肿扭伤骨科",
        "message": "脚脖子肿，昨天扭了",
        "expect_route": "symptom",
        "expect_dept": "骨科",
    },
    {
        "name": "急诊踝肿",
        "message": "脚脖子肿，不能动，皮发紫",
        "expect_route": "symptom",
        "expect_dept": "急诊",
    },
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()
    base = args.base_url.rstrip("/")

    health = requests.get(f"{base}/healthz", timeout=10)
    health.raise_for_status()
    print(f"[OK] healthz: {health.json()}")

    user_id = f"test-{uuid.uuid4().hex[:8]}"
    requests.post(f"{base}/users", json={"user_id": user_id}, timeout=10)

    passed = 0
    for case in CASES:
        thread_resp = requests.post(
            f"{base}/threads",
            json={"user_id": user_id, "title": case["name"]},
            timeout=10,
        )
        thread_resp.raise_for_status()
        thread_id = thread_resp.json()["thread_id"]

        resp = requests.post(
            f"{base}/chat",
            json={"user_id": user_id, "thread_id": thread_id, "message": case["message"]},
            timeout=args.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        route = (data.get("intent_result") or {}).get("triage_route")
        reply = data.get("reply", "")

        payload = {
            "case": case["name"],
            "message": case["message"],
            "triage_route": route,
            "reply_preview": reply[:120],
            "intent_result": data.get("intent_result"),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))

        expected_route = case["expect_route"]
        if expected_route == "reject":
            ok = route in (expected_route, None)
        else:
            ok = route == expected_route
        if case.get("expect_reply"):
            ok = ok and reply.strip() == case["expect_reply"]
        if case.get("expect_dept"):
            ok = ok and case["expect_dept"] in reply
        if ok:
            passed += 1
            print(f"[OK] {case['name']}\n")
        else:
            print(f"[FAIL] {case['name']}: expected route={case['expect_route']}\n")

    print(f"Result: {passed}/{len(CASES)} passed")
    if passed != len(CASES):
        sys.exit(1)


if __name__ == "__main__":
    main()
