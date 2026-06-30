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
