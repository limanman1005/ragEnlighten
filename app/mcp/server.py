"""MCP server exposing shared Agent retrieval tools."""

from __future__ import annotations

from app.core.config import settings
from app.services.agent_tools import (
    CollectionOverviewRequest,
    KnowledgeBaseSearchRequest,
    WebSearchToolRequest,
    run_collection_overview,
    run_knowledge_base_search,
    run_web_search_tool,
)

try:
    from mcp.server.fastmcp import FastMCP
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
    raise RuntimeError("Install the `mcp` package to run the Agent MCP server.") from exc

mcp = FastMCP("rag-enlighten-tools", json_response=True)


@mcp.tool()
def knowledge_base_search(
    query: str,
    collection_name: str | None = None,
    source_types: list[str] | None = None,
    top_k: int | None = None,
    timeout_seconds: float | None = None,
    max_retries: int = 0,
) -> dict:
    """Search the indexed knowledge base for evidence relevant to the user's question."""
    try:
        request = KnowledgeBaseSearchRequest(
            query=query,
            collection_name=collection_name,
            source_types=source_types,
            top_k=top_k,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
        )
    except Exception as exc:
        from app.services.agent_tools import validation_error_result

        return validation_error_result(str(exc)).to_tool_payload()

    return run_knowledge_base_search(request).to_tool_payload()


@mcp.tool()
def collection_overview(
    collection_name: str | None = None,
    timeout_seconds: float | None = None,
    max_retries: int = 0,
) -> dict:
    """Return internal metadata about configured models and available collections."""
    request = CollectionOverviewRequest(
        collection_name=collection_name,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
    )
    return run_collection_overview(request).to_tool_payload()


@mcp.tool()
def web_search(
    query: str,
    top_k: int | None = None,
    timeout_seconds: float | None = None,
    max_retries: int = 0,
) -> dict:
    """Search the web for external or current information beyond the indexed knowledge base."""
    try:
        request = WebSearchToolRequest(
            query=query,
            top_k=top_k,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
        )
    except Exception as exc:
        from app.services.agent_tools import validation_error_result

        return validation_error_result(str(exc)).to_tool_payload()

    return run_web_search_tool(request).to_tool_payload()


def main() -> None:
    mcp.run(
        transport="streamable-http",
        host=settings.mcp_server_host,
        port=settings.mcp_server_port,
    )


if __name__ == "__main__":
    main()
