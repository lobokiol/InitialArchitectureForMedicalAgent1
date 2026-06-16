from __future__ import annotations

import uuid

import requests


def run_live_smoke(
    cases: list[dict],
    base_url: str = "http://127.0.0.1:8000",
    timeout: float = 180.0,
) -> tuple[list[dict], int, int]:
    base = base_url.rstrip("/")
    requests.get(f"{base}/healthz", timeout=10).raise_for_status()
    user_id = f"foot-eval-{uuid.uuid4().hex[:8]}"
    requests.post(f"{base}/users", json={"user_id": user_id}, timeout=10)

    results: list[dict] = []
    passed = 0
    for c in cases:
        subset = c.get("subset")
        expect_dept = c.get("expect_dept")
        expect_route = c.get("expect_route")
        expect_emergency = c.get("expect_emergency")

        tr = requests.post(
            f"{base}/threads",
            json={"user_id": user_id, "title": c["id"]},
            timeout=10,
        )
        tr.raise_for_status()
        thread_id = tr.json()["thread_id"]
        resp = requests.post(
            f"{base}/chat",
            json={
                "user_id": user_id,
                "thread_id": thread_id,
                "message": c.get("message") or "",
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        reply = data.get("reply", "")
        route = (data.get("intent_result") or {}).get("triage_route")

        ok = True
        if subset == "D":
            from app.domain.triage_intent import REJECT_MESSAGE

            ok = reply.strip() == REJECT_MESSAGE
        elif subset == "E" or expect_emergency:
            ok = "急诊" in reply
        elif expect_dept:
            ok = expect_dept in reply
        elif expect_route:
            ok = route == expect_route

        if ok:
            passed += 1
        results.append(
            {
                "id": c["id"],
                "subset": subset,
                "expect_dept": expect_dept,
                "route": route,
                "ok": ok,
                "reply_preview": reply[:120],
            }
        )
        mark = "OK" if ok else "FAIL"
        print(f"[LIVE {mark}] {c['id']} subset={subset}")

    return results, passed, len(cases)
