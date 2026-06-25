"""Quick LangSmith connectivity check.

  .\\.venv\\Scripts\\python.exe tests\\test_langsmith.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core import config  # noqa: F401 — loads project-root .env

from langsmith import Client

key = os.getenv("LANGSMITH_API_KEY") or ""
endpoint = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
project = os.getenv("LANGSMITH_PROJECT")

print("endpoint=", endpoint)
print("project=", project)
print("tracing=", os.getenv("LANGCHAIN_TRACING_V2"), os.getenv("LANGSMITH_TRACING"))
print("api_key_set=", bool(key), "prefix=", key[:12] + "..." if len(key) > 12 else "(empty)")

if not key:
    print("\n[FAIL] LANGSMITH_API_KEY is empty. Set it in project-root .env")
    sys.exit(1)

c = Client(api_key=key, api_url=endpoint)

try:
    info = c.info
    print("api_ok=", True, "tenant=", getattr(info, "tenant_id", None))
except Exception as exc:
    err = str(exc)
    print("api_ok=", False, "error=", err[:200])
    if "403" in err or "Forbidden" in err:
        print(
            "\n[403 Forbidden] Key is rejected or lacks permission. Fix:\n"
            "  1. smith.langchain.com → Settings → API Keys → Create new key\n"
            "  2. Use a Personal API Key (full access), not a restricted service key\n"
            "  3. Paste new key into .env as LANGSMITH_API_KEY=...\n"
            "  4. If account is EU/APAC, set LANGSMITH_ENDPOINT to the region URL\n"
            "  5. Revoke old keys that were exposed in chat"
        )
    elif "401" in err:
        print("\n[401] Invalid API key — regenerate at smith.langchain.com")
    sys.exit(1)

try:
    names = [p.name for p in c.list_projects(limit=5)]
    print("projects=", names)
    if project and project not in names:
        print(f"[note] '{project}' not in list yet — auto-created on first trace")
except Exception as exc:
    err = str(exc)
    print("list_projects=", "403 (read restricted)" if "403" in err else "failed")
    if "403" in err:
        print(
            "  Service/restricted keys often cannot list projects but CAN still ingest traces."
        )

# Verify write path (POST /runs — what LangGraph tracing needs)
try:
    c.create_run(
        name="langsmith_smoke_test",
        run_type="chain",
        inputs={"check": "connectivity"},
        outputs={"check": "ok"},
        project_name=project,
    )
    print("trace_write=", "OK")
except Exception as exc:
    err = str(exc)
    print("trace_write=", "FAIL (403)" if "403" in err else "FAIL")
    print(
        "\n[FAIL] Current key cannot upload traces.\n"
        "  Your lsv2_sk_* key is likely a restricted Service Key or was revoked.\n"
        "  Fix: smith.langchain.com → Settings → API Keys → Create **Personal API Key**\n"
        "       Update LANGSMITH_API_KEY in .env → re-run this script."
    )
    sys.exit(1)

print(
    "\n[OK] Key works for tracing. Restart API → send /chat → LangSmith → "
    f"{project or 'default'} → Tracing (refresh after ~10s)."
)
