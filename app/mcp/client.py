from __future__ import annotations

import asyncio
import json
import shlex
import time
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from app.core import config
from app.core.logging import logger
from app.domain.models import DepartmentIntro, DepartmentRoute, OnCallDoctor

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

_FOLLOWUP_TOOLS = frozenset({"get_department_intro", "get_department_route"})

_tools_cache: tuple[float, list[Any]] | None = None


def _server_params() -> StdioServerParameters:
    parts = shlex.split(config.MCP_SERVER_COMMAND, posix=False)
    command = parts[0]
    args = parts[1:]
    return StdioServerParameters(
        command=command,
        args=args,
        cwd=str(_PROJECT_ROOT),
    )


def _parse_tool_text(result: Any) -> Any:
    if not result.content:
        return None
    raw = result.content[0].text
    if not raw:
        return None
    return json.loads(raw)


class McpHospitalClient:
    async def list_tools(self, *, followup_only: bool = False) -> list[Any]:
        global _tools_cache
        now = time.monotonic()
        if _tools_cache and now - _tools_cache[0] < 300:
            tools = _tools_cache[1]
        else:
            params = _server_params()
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await asyncio.wait_for(session.initialize(), timeout=config.MCP_TIMEOUT_SECONDS)
                    listed = await asyncio.wait_for(session.list_tools(), timeout=config.MCP_TIMEOUT_SECONDS)
            tools = list(listed.tools)
            _tools_cache = (now, tools)
        if not followup_only:
            return tools
        return [t for t in tools if t.name in _FOLLOWUP_TOOLS]

    async def call_tool(self, name: str, arguments: dict) -> Any:
        params = _server_params()
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await asyncio.wait_for(session.initialize(), timeout=config.MCP_TIMEOUT_SECONDS)
                result = await asyncio.wait_for(
                    session.call_tool(name, arguments),
                    timeout=config.MCP_TIMEOUT_SECONDS,
                )
        return _parse_tool_text(result)


_client = McpHospitalClient()


def get_mcp_client() -> McpHospitalClient:
    return _client


async def fetch_oncall_appointments(department: str) -> list[OnCallDoctor]:
    data = await _client.call_tool("get_oncall_appointments", {"department": department})
    if not data:
        return []
    return [OnCallDoctor.model_validate(item) for item in data]


async def fetch_department_intro(department: str) -> DepartmentIntro | dict:
    data = await _client.call_tool("get_department_intro", {"department": department})
    if isinstance(data, dict) and data.get("error"):
        return data
    return DepartmentIntro.model_validate(data)


async def fetch_department_route(department: str, from_location: str = "导诊台") -> DepartmentRoute | dict:
    data = await _client.call_tool(
        "get_department_route",
        {"department": department, "from_location": from_location},
    )
    if isinstance(data, dict) and data.get("error"):
        return data
    payload = dict(data)
    if "from" in payload and "from_location" not in payload:
        payload["from_location"] = payload.pop("from")
    return DepartmentRoute.model_validate(payload)


def fetch_oncall_appointments_sync(department: str) -> list[OnCallDoctor]:
    try:
        return asyncio.run(fetch_oncall_appointments(department))
    except Exception:
        logger.exception("MCP fetch_oncall_appointments failed dept=%s", department)
        raise


def fetch_department_intro_sync(department: str) -> DepartmentIntro | dict:
    try:
        return asyncio.run(fetch_department_intro(department))
    except Exception:
        logger.exception("MCP fetch_department_intro failed dept=%s", department)
        raise


def fetch_department_route_sync(department: str, from_location: str = "导诊台") -> DepartmentRoute | dict:
    try:
        return asyncio.run(fetch_department_route(department, from_location))
    except Exception:
        logger.exception("MCP fetch_department_route failed dept=%s", department)
        raise
