"""Quick E2E chat test against local API."""
import json
import sys
import urllib.error
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
UID = "demo-verify2"
SYMPTOM = "肚子疼"


def req(method: str, path: str, body=None):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(
        BASE + path,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(r, timeout=120) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def main():
    st, _ = req("POST", "/users", {"user_id": UID, "name": "e2e"})
    print("users", st)
    _, cur = req("GET", f"/threads/current?user_id={UID}")
    tid = cur["thread_id"]
    msg = SYMPTOM

    for step in range(12):
        st, data = req("POST", "/chat", {"user_id": UID, "thread_id": tid, "message": msg})
        if st != 200:
            print("FAIL", st, data)
            return 1
        print(
            f"step {step}: clarify={data.get('awaiting_clarify')} "
            f"dept={data.get('awaiting_dept_choice')} multi={data.get('multi_select')}"
        )
        if data.get("locked_department"):
            print("DEPT:", data["locked_department"], "conf:", data.get("dept_confidence"))
        tid = data["thread_id"]

        if data.get("awaiting_clarify") and data.get("clarify_choices"):
            msg = data["clarify_choices"][0]["label"]
            print("  pick clarify:", msg)
            continue
        if data.get("awaiting_dept_choice") and data.get("dept_choices"):
            if data.get("multi_select"):
                msg = "5"  # 都没有
            else:
                msg = data["dept_choices"][0]["label"]
            print("  pick dept:", msg)
            continue

        print("FINAL locked:", data.get("locked_department"))
        print("FINAL conf:", data.get("dept_confidence"), data.get("dept_confidence_passed"))
        print("reply:", data.get("reply", "")[:500])
        return 0

    print("max steps exceeded")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
