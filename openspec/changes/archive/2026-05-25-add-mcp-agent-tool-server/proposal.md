## Why

The current ReAct Agent uses in-process LangChain tools for knowledge-base retrieval, collection metadata, and web search. To make the tool layer reusable by MCP clients while preserving stable local behavior, the Agent needs a dual-path tool architecture with structured inputs, structured outputs, and failure containment.

## What Changes

- Add an MCP tool server that exposes the Agent retrieval tools through MCP without removing the existing FastAPI chat endpoints.
- Add a configurable React Agent MCP client path controlled by an `MCP_ENABLED` switch; when disabled, the Agent keeps using local in-process tools.
- Refactor tool execution behind shared service functions so local LangChain tools and MCP tools return the same structured result shape.
- Extend knowledge-base retrieval tool inputs to support `collection_name`, `source_types`, `top_k`, `timeout_seconds`, and `max_retries`.
- Standardize tool results with content, status, error details, and source metadata so `QueryResponse.sources`, traces, and grounding validation remain explainable.
- Add empty-result fallback and retry containment for supported Agent tools.
- Exclude permission-scope or ACL-based retrieval filtering from this change.

## Capabilities

### New Capabilities

- `agent-mcp-tool-server`: Exposes Agent retrieval and metadata tools through MCP and allows the React Agent to consume them through a configurable MCP client path.

### Modified Capabilities

- `agent-web-search-tool`: Standardizes web search tool input/output behavior with the shared Agent tool schema and retry/failure containment used by the MCP-capable tool layer.

## Impact

- Affected code: `app/services/react_agent.py`, new shared Agent tool service modules, new MCP server/client modules, `app/core/config.py`, `.env.example`, `requirements.txt`, and focused tests under `tests/`.
- API impact: existing FastAPI endpoint paths and public `QueryResponse` shape remain unchanged.
- Dependency impact: adds Python MCP SDK and LangChain MCP adapter dependencies.
- Operational impact: MCP mode requires running or mounting the MCP tool server and configuring the React Agent MCP client URL; local tool mode remains the default fallback.
