"""Quick E2E chat test against local API."""
import json
import sys
import urllib.error
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
PHONE = "13900000088"
PASSWORD = "eval-pass-123"
SYMPTOM = "肚子疼"


def req(method: str, path: str, body=None, headers=None):
    hdrs = {"Content-Type": "application/json"}
    if headers:
        hdrs.update(headers)
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(BASE + path, data=data, method=method, headers=hdrs)
    try:
        with urllib.request.urlopen(r, timeout=120) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body_text = e.read().decode()
        try:
            return e.code, json.loads(body_text)
        except json.JSONDecodeError:
            return e.code, body_text


def auth_token() -> str:
    st, data = req("POST", "/auth/login", {"phone": PHONE, "password": PASSWORD})
    if st == 200:
        return data["access_token"]
    st, data = req("POST", "/auth/register", {"phone": PHONE, "password": PASSWORD, "display_name": "e2e"})
    if st != 200:
        raise RuntimeError(f"auth failed: {st} {data}")
    return data["access_token"]


def main():
    token = auth_token()
    headers = {"Authorization": f"Bearer {token}"}

    _, cur = req("GET", "/threads/current", headers=headers)
    tid = cur["thread_id"]
    msg = SYMPTOM

    for step in range(12):
        st, data = req("POST", "/chat", {"thread_id": tid, "message": msg}, headers=headers)
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
                msg = "5"
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
