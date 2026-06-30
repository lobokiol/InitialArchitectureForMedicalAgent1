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


def _clear_oncall_patch() -> dict:
    return {"oncall_appointments": [], "oncall_fetch_error": None}


def fetch_oncall_node(state: AppState) -> dict:
    logger.info(">>> Enter node: fetch_oncall")
    if not config.MCP_ENABLED or not should_fetch(state):
        return _clear_oncall_patch()
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
