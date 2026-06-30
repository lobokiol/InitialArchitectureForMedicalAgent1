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
    parts = shlex.split(config.MCP_SERVER_COMMAND, posix=False)
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
    if not raw:
        return []
    data = json.loads(raw)
    return [OnCallDoctor.model_validate(item) for item in data]


def fetch_oncall_appointments_sync(department: str) -> list[OnCallDoctor]:
    try:
        return asyncio.run(fetch_oncall_appointments(department))
    except Exception:
        logger.exception("MCP fetch_oncall_appointments failed dept=%s", department)
        raise
