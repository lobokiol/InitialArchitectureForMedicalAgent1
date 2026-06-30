# HIS 值班医生预约 MCP 集成 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** After department recommendation succeeds, automatically call a hospital MCP tool to fetch on-call doctor appointments and return them to the frontend as independent cards with disabled「预约」buttons.

**Architecture:** Mock MCP Server (`mcp_server/`) exposes `get_oncall_appointments` over stdio. Local backend wraps it in `app/mcp/client.py`. A new LangGraph node `fetch_oncall` runs before `answer_generate` on all paths; it no-ops unless trigger conditions match. `ChatResponse` gains `oncall_appointments`; frontend renders `AppointmentCards` below the assistant bubble.

**Tech Stack:** Python 3.10+, FastAPI, LangGraph, Pydantic v2, `mcp` Python SDK (stdio), React + TypeScript + Tailwind (front_Web), pytest.

## Global Constraints

- Trigger: symptom chain (`locked_department` + `dept_confidence_passed=true`) OR disease chain (`disease_dept_result.departments` non-empty); **exclude 急诊**
- Department param: symptom → `locked_department`; disease → `departments[0].dept` (or `.department`)
- Data shape: `{name: str, time: str, slots: int}` × 3 doctors
- Appointments are **not** appended to `reply` Markdown; separate API field + UI cards
-「预约」button: `disabled`, no click handler
- `slots === 0`: show「已满」, button greyed out
- MCP failure must **not** block triage; set `oncall_fetch_error="暂无法获取预约信息"`, hide cards
- `MCP_ENABLED=false`: skip silently
- `MCP_TIMEOUT_SECONDS`: `5.0`
- Default `MCP_SERVER_COMMAND`: `python mcp_server/server.py` (parsed via `shlex.split`)

**Spec:** `docs/superpowers/specs/2026-06-30-his-oncall-mcp-design.md`

---

## File Map

| File | Responsibility |
|------|----------------|
| `app/domain/models.py` | `OnCallDoctor` model; `AppState.oncall_appointments`, `oncall_fetch_error` |
| `app/core/config.py` | `MCP_ENABLED`, `MCP_SERVER_COMMAND`, `MCP_TIMEOUT_SECONDS` |
| `mcp_server/mock_data.py` | Hardcoded 3-doctor JSON per department |
| `mcp_server/server.py` | Mock MCP Server (stdio), tool `get_oncall_appointments` |
| `app/mcp/client.py` | Async MCP client + sync wrapper for graph node |
| `app/graph/nodes/fetch_oncall.py` | `should_fetch`, `resolve_department`, `fetch_oncall_node` |
| `app/graph/builder.py` | Insert `fetch_oncall` before `answer_generate` on all paths |
| `app/api/routers/chat.py` | `ChatResponse.oncall_appointments`, `oncall_fetch_error` |
| `app/services/chat_service.py` | Pass through new fields from `AppState` |
| `requirements.txt` | Add `mcp>=1.0.0` |
| `tests/test_fetch_oncall.py` | Unit tests for trigger + department resolution |
| `front_Web/src/types/index.ts` | `OnCallDoctor`, extend `ChatResponse` |
| `front_Web/src/components/AppointmentCards.tsx` | Doctor cards UI |
| `front_Web/src/components/ChatStage.tsx` | Mount cards below assistant bubble |

---

### Task 1: Domain model `OnCallDoctor`

**Files:**
- Modify: `app/domain/models.py`
- Test: `tests/test_fetch_oncall.py` (model import only in this task)

**Interfaces:**
- Produces: `OnCallDoctor(name: str, time: str, slots: int)`
- Produces on `AppState`: `oncall_appointments: list[OnCallDoctor]`, `oncall_fetch_error: str | None = None`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_fetch_oncall.py
from app.domain.models import OnCallDoctor

def test_oncall_doctor_model():
    doc = OnCallDoctor(name="张医生", time="14:00-18:00", slots=3)
    assert doc.name == "张医生"
    assert doc.slots == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_fetch_oncall.py::test_oncall_doctor_model -v`
Expected: FAIL — `OnCallDoctor` not defined

- [ ] **Step 3: Add model to `app/domain/models.py`**

After `RetrievedDoc`, add:

```python
class OnCallDoctor(BaseModel):
    name: str
    time: str
    slots: int

    model_config = ConfigDict(extra="ignore")
```

On `AppState`, after `dept_confidence_passed`, add:

```python
oncall_appointments: list[OnCallDoctor] = Field(default_factory=list)
oncall_fetch_error: str | None = None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_fetch_oncall.py::test_oncall_doctor_model -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/domain/models.py tests/test_fetch_oncall.py
git commit -m "feat: add OnCallDoctor model and AppState fields"
```

---

### Task 2: Mock MCP Server

**Files:**
- Create: `mcp_server/mock_data.py`
- Create: `mcp_server/server.py`
- Modify: `requirements.txt`

**Interfaces:**
- Produces: stdio MCP server registering tool `get_oncall_appointments(department: str) -> str` (JSON array string)

- [ ] **Step 1: Add dependency**

In `requirements.txt`, append:

```
mcp>=1.0.0
```

Run: `pip install mcp>=1.0.0`

- [ ] **Step 2: Create `mcp_server/mock_data.py`**

```python
from __future__ import annotations

BASE_DOCTORS = [
    {"name": "张医生", "time": "14:00-18:00", "slots": 3},
    {"name": "李医生", "time": "08:00-12:00", "slots": 5},
    {"name": "王医生", "time": "全天", "slots": 0},
]


def doctors_for_department(department: str) -> list[dict]:
    prefix = department.strip() or "综合"
    return [
        {**doc, "name": f"{prefix}·{doc['name']}"}
        for doc in BASE_DOCTORS
    ]
```

- [ ] **Step 3: Create `mcp_server/server.py`**

```python
from __future__ import annotations

import json

from mcp.server.fastmcp import FastMCP

from mock_data import doctors_for_department

mcp = FastMCP("hospital-his-mock")


@mcp.tool()
def get_oncall_appointments(department: str) -> str:
    """查询指定科室值班医生预约信息。"""
    doctors = doctors_for_department(department)
    return json.dumps(doctors, ensure_ascii=False)


if __name__ == "__main__":
    mcp.run(transport="stdio")
```

- [ ] **Step 4: Smoke-test server manually**

Run from repo root:

```bash
python mcp_server/server.py
```

Expected: process starts and waits on stdin (Ctrl+C to exit). No import errors.

- [ ] **Step 5: Commit**

```bash
git add mcp_server/mock_data.py mcp_server/server.py requirements.txt
git commit -m "feat: add mock MCP server for on-call appointments"
```

---

### Task 3: MCP Client

**Files:**
- Create: `app/mcp/__init__.py`
- Create: `app/mcp/client.py`
- Modify: `app/core/config.py`
- Test: `tests/test_fetch_oncall.py`

**Interfaces:**
- Consumes: `OnCallDoctor` from `app.domain.models`
- Consumes: `MCP_ENABLED`, `MCP_SERVER_COMMAND`, `MCP_TIMEOUT_SECONDS` from config
- Produces: `async def fetch_oncall_appointments(department: str) -> list[OnCallDoctor]`
- Produces: `def fetch_oncall_appointments_sync(department: str) -> list[OnCallDoctor]`

- [ ] **Step 1: Add config**

At end of `app/core/config.py`:

```python
MCP_ENABLED: bool = os.getenv("MCP_ENABLED", "true").lower() in ("1", "true", "yes")
MCP_SERVER_COMMAND: str = os.getenv("MCP_SERVER_COMMAND", "python mcp_server/server.py")
MCP_TIMEOUT_SECONDS: float = float(os.getenv("MCP_TIMEOUT_SECONDS", "5.0"))
```

- [ ] **Step 2: Create `app/mcp/__init__.py`**

```python
from app.mcp.client import fetch_oncall_appointments_sync

__all__ = ["fetch_oncall_appointments_sync"]
```

- [ ] **Step 3: Create `app/mcp/client.py`**

```python
from __future__ import annotations

import asyncio
import json
import shlex
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from app.core import config
from app.core.logging import logger
from app.domain.models import OnCallDoctor

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _server_params() -> StdioServerParameters:
    parts = shlex.split(config.MCP_SERVER_COMMAND)
    command = parts[0]
    args = parts[1:]
    return StdioServerParameters(
        command=command,
        args=args,
        cwd=str(_PROJECT_ROOT),
    )


async def fetch_oncall_appointments(department: str) -> list[OnCallDoctor]:
    params = _server_params()
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await asyncio.wait_for(session.initialize(), timeout=config.MCP_TIMEOUT_SECONDS)
            result = await asyncio.wait_for(
                session.call_tool("get_oncall_appointments", {"department": department}),
                timeout=config.MCP_TIMEOUT_SECONDS,
            )
    if not result.content:
        return []
    raw = result.content[0].text
    data = json.loads(raw)
    return [OnCallDoctor.model_validate(item) for item in data]


def fetch_oncall_appointments_sync(department: str) -> list[OnCallDoctor]:
    return asyncio.run(fetch_oncall_appointments(department))
```

- [ ] **Step 4: Write integration test (skipped if MCP disabled)**

Append to `tests/test_fetch_oncall.py`:

```python
import pytest
from app.core import config
from app.mcp.client import fetch_oncall_appointments_sync

@pytest.mark.skipif(not config.MCP_ENABLED, reason="MCP disabled")
def test_mcp_client_returns_three_doctors():
    doctors = fetch_oncall_appointments_sync("骨科")
    assert len(doctors) == 3
    assert doctors[0].name.startswith("骨科")
```

- [ ] **Step 5: Run test**

Run: `python -m pytest tests/test_fetch_oncall.py::test_mcp_client_returns_three_doctors -v`
Expected: PASS (spawns mock server subprocess)

- [ ] **Step 6: Commit**

```bash
git add app/mcp/ app/core/config.py tests/test_fetch_oncall.py
git commit -m "feat: add MCP client for on-call appointment lookup"
```

---

### Task 4: `fetch_oncall` graph node helpers

**Files:**
- Create: `app/graph/nodes/fetch_oncall.py`
- Test: `tests/test_fetch_oncall.py`

**Interfaces:**
- Consumes: `AppState`, `OnCallDoctor`, `DiseaseDeptResult`
- Produces: `should_fetch(state: AppState) -> bool`
- Produces: `resolve_department(state: AppState) -> str | None`
- Produces: `fetch_oncall_node(state: AppState) -> dict`

- [ ] **Step 1: Write failing tests**

```python
from app.domain.models import AppState, DiseaseDeptResult
from app.graph.nodes.fetch_oncall import resolve_department, should_fetch

def test_should_fetch_symptom_chain():
    state = AppState(locked_department="骨科", dept_confidence_passed=True)
    assert should_fetch(state) is True

def test_should_fetch_skips_emergency():
    state = AppState(locked_department="急诊", dept_confidence_passed=True)
    assert should_fetch(state) is False

def test_should_fetch_disease_chain():
    state = AppState(
        disease_dept_result=DiseaseDeptResult(
            diseases=["骨折"],
            departments=[{"dept": "骨科"}],
        )
    )
    assert should_fetch(state) is True

def test_resolve_department_symptom():
    state = AppState(locked_department="骨科")
    assert resolve_department(state) == "骨科"

def test_resolve_department_disease_first():
    state = AppState(
        disease_dept_result=DiseaseDeptResult(
            diseases=["骨折"],
            departments=[{"dept": "骨科"}, {"dept": "康复科"}],
        )
    )
    assert resolve_department(state) == "骨科"
```

- [ ] **Step 2: Run tests — expect FAIL**

Run: `python -m pytest tests/test_fetch_oncall.py -k "should_fetch or resolve_department" -v`

- [ ] **Step 3: Implement `app/graph/nodes/fetch_oncall.py`**

```python
from app.core import config
from app.core.logging import logger
from app.domain.models import AppState
from app.mcp.client import fetch_oncall_appointments_sync


def should_fetch(state: AppState) -> bool:
    if state.locked_department == "急诊":
        return False
    if state.locked_department and state.dept_confidence_passed is True:
        return True
    ddr = state.disease_dept_result
    if ddr and ddr.departments:
        return True
    return False


def resolve_department(state: AppState) -> str | None:
    if state.locked_department:
        return state.locked_department
    ddr = state.disease_dept_result
    if ddr and ddr.departments:
        first = ddr.departments[0]
        if isinstance(first, dict):
            return first.get("dept") or first.get("department")
        return str(first) if first else None
    return None


def fetch_oncall_node(state: AppState) -> dict:
    logger.info(">>> Enter node: fetch_oncall")
    if not config.MCP_ENABLED or not should_fetch(state):
        return {}
    dept = resolve_department(state)
    if not dept:
        return {}
    try:
        doctors = fetch_oncall_appointments_sync(dept)
        if not doctors:
            return {
                "oncall_appointments": [],
                "oncall_fetch_error": "暂无法获取预约信息",
            }
        return {
            "oncall_appointments": doctors,
            "tool_call_result": {"department": dept, "doctors": [d.model_dump() for d in doctors]},
        }
    except Exception:
        logger.exception("fetch_oncall failed dept=%s", dept)
        return {
            "oncall_appointments": [],
            "oncall_fetch_error": "暂无法获取预约信息",
        }
```

- [ ] **Step 4: Run tests — expect PASS**

Run: `python -m pytest tests/test_fetch_oncall.py -v`

- [ ] **Step 5: Commit**

```bash
git add app/graph/nodes/fetch_oncall.py tests/test_fetch_oncall.py
git commit -m "feat: add fetch_oncall LangGraph node with trigger helpers"
```

---

### Task 5: Wire graph

**Files:**
- Modify: `app/graph/builder.py`

**Interfaces:**
- Consumes: `fetch_oncall_node` from `app.graph.nodes.fetch_oncall`
- All former edges targeting `answer_generate` now target `fetch_oncall`; `fetch_oncall` → `answer_generate`

- [ ] **Step 1: Update `app/graph/builder.py`**

Add import:

```python
from app.graph.nodes.fetch_oncall import fetch_oncall_node
```

Add node:

```python
graph.add_node("fetch_oncall", fetch_oncall_node)
```

Change edges:

```python
# was: graph.add_edge("disease_dept", "answer_generate")
graph.add_edge("disease_dept", "fetch_oncall")

# was: "answer_generate": "answer_generate" in symptom_clarify conditional
"answer_generate": "fetch_oncall",

# was: dept_rules conditional "dept_confidence": "dept_confidence"  (unchanged)
# was: dept_disambiguation conditional "answer_generate": "answer_generate"
"answer_generate": "fetch_oncall",

# was: dept_confidence conditional "answer_generate": "answer_generate"
"answer_generate": "fetch_oncall",

graph.add_edge("fetch_oncall", "answer_generate")
```

- [ ] **Step 2: Verify graph compiles**

Run: `python -c "from app.graph.builder import build_app; from langgraph.checkpoint.memory import MemorySaver; build_app(MemorySaver())"`
Expected: no import / wiring errors

- [ ] **Step 3: Commit**

```bash
git add app/graph/builder.py
git commit -m "feat: wire fetch_oncall node before answer_generate"
```

---

### Task 6: API response passthrough

**Files:**
- Modify: `app/api/routers/chat.py`
- Modify: `app/services/chat_service.py`

**Interfaces:**
- Consumes: `OnCallDoctor` from `app.domain.models`
- Produces on `ChatResponse`: `oncall_appointments: List[OnCallDoctor]`, `oncall_fetch_error: Optional[str]`

- [ ] **Step 1: Extend `ChatResponse` in `app/api/routers/chat.py`**

Add import:

```python
from app.domain.models import IntentResult, RetrievedDoc, OnCallDoctor
```

Add fields to `ChatResponse`:

```python
oncall_appointments: List[OnCallDoctor] = []
oncall_fetch_error: Optional[str] = None
```

- [ ] **Step 2: Pass through in `chat_service.chat_once` return dict**

```python
"oncall_appointments": [d.model_dump() for d in state.oncall_appointments],
"oncall_fetch_error": state.oncall_fetch_error,
```

- [ ] **Step 3: Smoke-test import**

Run: `python -c "from app.api.routers.chat import ChatResponse; print(ChatResponse.model_fields.keys())"`
Expected: includes `oncall_appointments`, `oncall_fetch_error`

- [ ] **Step 4: Commit**

```bash
git add app/api/routers/chat.py app/services/chat_service.py
git commit -m "feat: expose oncall appointments in ChatResponse"
```

---

### Task 7: Frontend types and `AppointmentCards`

**Files:**
- Modify: `front_Web/src/types/index.ts`
- Create: `front_Web/src/components/AppointmentCards.tsx`
- Modify: `front_Web/src/components/ChatStage.tsx`

**Interfaces:**
- Consumes: `OnCallDoctor[]` from `turn.chatSnapshot?.oncall_appointments`

- [ ] **Step 1: Extend types in `front_Web/src/types/index.ts`**

```typescript
export interface OnCallDoctor {
  name: string;
  time: string;
  slots: number;
}

// inside ChatResponse:
oncall_appointments?: OnCallDoctor[];
oncall_fetch_error?: string;
```

- [ ] **Step 2: Create `front_Web/src/components/AppointmentCards.tsx`**

```tsx
import type { OnCallDoctor } from '../types';

interface AppointmentCardsProps {
  doctors: OnCallDoctor[];
}

export function AppointmentCards({ doctors }: AppointmentCardsProps) {
  if (!doctors.length) return null;

  return (
    <div className="w-full max-w-lg mx-auto">
      <p className="text-xs font-medium text-brand-700 mb-2">值班医生预约</p>
      <div className="flex flex-col sm:flex-row gap-2">
        {doctors.map((doc) => {
          const full = doc.slots <= 0;
          return (
            <div
              key={doc.name}
              className="flex-1 rounded-xl border border-brand-500/20 bg-white px-3 py-3 shadow-sm"
            >
              <p className="font-medium text-gray-900 text-sm">{doc.name}</p>
              <p className="text-xs text-gray-500 mt-1">{doc.time}</p>
              <p className="text-xs text-brand-700 mt-1">{full ? '已满' : `余号 ${doc.slots}`}</p>
              <button
                type="button"
                disabled
                className="mt-2 w-full rounded-lg bg-brand-500/90 text-white text-xs py-1.5 disabled:opacity-40 disabled:cursor-not-allowed"
              >
                预约
              </button>
            </div>
          );
        })}
      </div>
    </div>
  );
}
```

- [ ] **Step 3: Mount in `ChatStage.tsx`**

Add import:

```tsx
import { AppointmentCards } from './AppointmentCards';
```

After assistant `MessageBubble`, before choice components:

```tsx
{turn.chatSnapshot?.oncall_appointments &&
  turn.chatSnapshot.oncall_appointments.length > 0 && (
    <AppointmentCards doctors={turn.chatSnapshot.oncall_appointments} />
  )}
```

- [ ] **Step 4: Typecheck frontend**

Run from `front_Web/`:

```bash
npm run build
```

Expected: build succeeds

- [ ] **Step 5: Commit**

```bash
git add front_Web/src/types/index.ts front_Web/src/components/AppointmentCards.tsx front_Web/src/components/ChatStage.tsx
git commit -m "feat: add AppointmentCards UI for on-call doctors"
```

---

### Task 8: End-to-end verification

**Files:** (no new files)

- [ ] **Step 1: Run backend unit tests**

Run: `python -m pytest tests/test_fetch_oncall.py -v`
Expected: all PASS

- [ ] **Step 2: Manual disease-chain smoke test**

With backend + dependencies running, send a disease-routed message (e.g. mention a disease in `disease_kb`). Confirm `/chat` JSON includes:

```json
"oncall_appointments": [
  {"name": "骨科·张医生", "time": "14:00-18:00", "slots": 3},
  ...
]
```

- [ ] **Step 3: Manual emergency skip**

Trigger 急诊 recommendation. Confirm `oncall_appointments` is `[]` and no cards in UI.

- [ ] **Step 4: MCP failure path**

Set `MCP_ENABLED=false` or break `MCP_SERVER_COMMAND`. Confirm triage reply still returns; `oncall_fetch_error` set or cards hidden.

- [ ] **Step 5: Update spec status**

In `docs/superpowers/specs/2026-06-30-his-oncall-mcp-design.md`, change `**状态**: 待实现` → `**状态**: 已实现`.

- [ ] **Step 6: Commit**

```bash
git add docs/superpowers/specs/2026-06-30-his-oncall-mcp-design.md
git commit -m "chore: mark HIS oncall MCP spec as implemented"
```

---

## Self-Review Checklist

| Spec requirement | Task |
|------------------|------|
| Mock MCP Server stdio | Task 2 |
| Tool `get_oncall_appointments(department)` | Task 2 |
| MCP Client with timeout 5s | Task 3 |
| Trigger symptom + disease, skip 急诊 | Task 4 |
| Dept param rules | Task 4 |
| Graph node before answer_generate | Task 5 |
| All paths to answer_generate via fetch_oncall | Task 5 |
| ChatResponse fields | Task 6 |
| Independent cards, disabled 预约 | Task 7 |
| Failure does not block triage | Task 4 |
| MCP_ENABLED=false skip | Task 4 |
| No DB | (no task — N/A) |

**Placeholder scan:** None found.

**Type consistency:** `OnCallDoctor` defined in Task 1; used in Tasks 3–7 with matching `{name, time, slots}`.
