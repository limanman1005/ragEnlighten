"""MCP client helpers for loading Agent tools into the ReAct Agent runtime."""

from __future__ import annotations

import logging
from typing import Any

from app.core.config import settings

logger = logging.getLogger("uvicorn.error")


async def load_mcp_tools() -> tuple[list[Any], str | None]:
    """Load MCP tools from the configured server.

    Returns a tuple of (tools, error_message). When loading fails, tools is empty
    and error_message explains why.
    """
    if not settings.mcp_enabled:
        return [], None

    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient
    except ModuleNotFoundError as exc:
        message = "langchain-mcp-adapters is not installed"
        logger.warning("[mcp_client.load] %s", message)
        return [], message

    client = MultiServerMCPClient(
        {
            "rag_tools": {
                "transport": "http",
                "url": settings.mcp_tools_url,
            }
        }
    )
    try:
        tools = await client.get_tools()
    except Exception as exc:
        message = f"Failed to load MCP tools from {settings.mcp_tools_url}: {exc}"
        logger.warning("[mcp_client.load] %s", message)
        return [], message

    if settings.mcp_tool_name_prefix:
        for tool in tools:
            name = getattr(tool, "name", None)
            if name and not name.startswith("rag_tools_"):
                tool.name = f"rag_tools_{name}"

    logger.info("[mcp_client.load] loaded_tools=%s url=%s", len(tools), settings.mcp_tools_url)
    return tools, None
