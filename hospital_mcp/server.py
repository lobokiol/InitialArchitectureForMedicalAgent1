from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow `python hospital_mcp/server.py` with sibling imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mcp.server.fastmcp import FastMCP

from adapters.mock_store import doctors_for_department, intro_for_department, route_for_department

mcp = FastMCP("hospital-his")


@mcp.tool()
def get_oncall_appointments(department: str) -> str:
    """查询指定科室值班医生预约信息。"""
    return json.dumps(doctors_for_department(department), ensure_ascii=False)


@mcp.tool()
def get_department_intro(department: str) -> str:
    """查询科室介绍、诊疗范围与就诊提示。"""
    return json.dumps(intro_for_department(department), ensure_ascii=False)


@mcp.tool()
def get_department_route(department: str, from_location: str = "导诊台") -> str:
    """查询从导诊台（或指定位置）到目标科室的步行路线。"""
    return json.dumps(route_for_department(department, from_location), ensure_ascii=False)


if __name__ == "__main__":
    mcp.run(transport="stdio")
