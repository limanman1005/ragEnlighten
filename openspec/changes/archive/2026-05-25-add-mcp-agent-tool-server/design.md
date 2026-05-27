## Context

The project currently exposes Agentic RAG through FastAPI endpoints and uses LangChain/LangGraph in-process tools for knowledge-base retrieval, collection metadata, and web search. The ReAct Agent is the most natural first integration point because it already passes a tool list to `create_agent`, while the plan-execute Agent still uses a fixed planning whitelist.

The tool layer also has hidden coupling: retrieval tools return text to the model while mutating an in-memory `documents` list so `QueryResponse.sources` and grounding validation can be populated. Moving tool execution behind MCP must preserve that source capture behavior, or answers may be generated with tool evidence while the API response appears ungrounded.

## Goals / Non-Goals

**Goals:**

- Provide a shared Agent tool service layer used by both local tools and MCP-exposed tools.
- Add an MCP server that exposes knowledge-base search, collection overview, and web search tools.
- Add a React Agent MCP client mode controlled by configuration, while keeping local in-process tools as the default and fallback path.
- Add structured tool input and output contracts covering document-type filtering, result limits, retries, timeout configuration, status, error details, and source metadata.
- Preserve existing FastAPI endpoint routes and `QueryResponse` shape.

**Non-Goals:**

- Do not add permission-scope filtering, user identity, ACL enforcement, or tenant authorization.
- Do not replace the LangGraph `/query` pipeline with MCP tools.
- Do not make the plan-execute Agent dynamically plan against arbitrary MCP tools in this change.
- Do not remove the existing deterministic mock web search provider.

## Decisions

### Decision 1: Use a shared tool service layer beneath local and MCP tools

The implementation will extract tool behavior into shared service functions that accept typed request objects and return typed result objects. Local LangChain `@tool` wrappers and MCP `@mcp.tool()` wrappers will call the same service functions.

Alternative considered: implement MCP tools separately from local tools. This would be faster initially but would duplicate retrieval formatting, source conversion, and failure handling, making local and MCP modes drift over time.

### Decision 2: Keep local tools as the default path and add MCP as an opt-in client path

`MCP_ENABLED=false` remains the default. When enabled, the ReAct Agent loads tools from the configured MCP server through a LangChain MCP adapter. If MCP loading fails, the system records a warning/debug event and falls back to local tools unless explicitly configured otherwise.

Alternative considered: fully migrate the ReAct Agent to MCP tools. That would better prove the MCP path but creates a larger operational dependency and higher risk for existing local development.

### Decision 3: Preserve source grounding through structured tool results

MCP tool results will include a model-facing `content` field and a machine-readable `sources` array. The React Agent will parse tool outputs and merge returned sources back into the existing `documents` collection so grounding validation and `QueryResponse.sources` continue to work.

Alternative considered: return only text from MCP tools. That is simpler but loses reliable source metadata and makes validation depend on parsing prose.

### Decision 4: Add document-type filtering without permission filtering

Knowledge-base search input will accept `source_types` such as `pdf`, `docx`, `txt`, `md`, `text`, or `web`. The vector-store query filter will combine `chunk_level=child` with `source_type` constraints when provided. Permission-scope filtering remains excluded because the project does not yet model user identity or ACL metadata.

Alternative considered: accept a generic metadata filter map. This is more flexible but easier to misuse and harder to validate. A first-class `source_types` field matches the requested scope and existing metadata.

### Decision 5: Bound retries and timeouts at the tool service boundary

Tool services will accept `timeout_seconds` and `max_retries`, with conservative bounds. Failures will return structured warning or error results rather than raising through Agent endpoints. Empty results return a successful no-results payload that does not count as grounded evidence.

Alternative considered: rely only on provider-specific retry behavior. Existing vector retrieval and collection metadata tools do not have a shared retry boundary, so this would leave inconsistent failure behavior.

## Risks / Trade-offs

- MCP adapter dependency changes could break import-time behavior -> Keep MCP imports lazy and make local mode work without MCP dependencies until MCP is enabled.
- MCP tool names may collide with local or external tools -> Support server-prefixed names and document expected tool names.
- Structured MCP tool output may be returned as JSON text by adapters -> Add a parser that accepts dictionaries, JSON strings, and simple text fallbacks.
- Timeout behavior around synchronous vector-store operations may not interrupt underlying blocking calls reliably -> Treat timeout as a configured execution budget for wrappers and provider calls where supported; always contain exceptions and expose elapsed/failure details.
- Plan-execute Agent will not consume MCP tools initially -> Keep the scope focused on React Agent and shared tool service extraction; a later change can make plan generation dynamic.

## Migration Plan

1. Add the shared tool request/result models and local service functions.
2. Update the ReAct Agent local tool wrappers to call the shared service layer.
3. Add MCP server wrappers around the same service functions.
4. Add React Agent MCP client loading behind `MCP_ENABLED`.
5. Update docs and `.env.example` with MCP settings and operational notes.
6. Verify local mode remains the default behavior and MCP mode can be enabled in tests with a mocked MCP client.

Rollback is to set `MCP_ENABLED=false`, which leaves the React Agent using local tools. If MCP dependencies or server startup fail, local mode remains available.

## Open Questions

- Should MCP server transport initially be a standalone `streamable-http` process or mounted into the existing FastAPI app? The implementation should prefer the least invasive option that is testable in this project.
- Should strict MCP mode fail closed when the MCP server is unavailable, or always fall back to local tools? The first implementation should default to fallback and leave strict mode as a future option.
