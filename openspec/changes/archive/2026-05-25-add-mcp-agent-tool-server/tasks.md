## 1. Shared Tool Contracts

- [x] 1.1 Add focused tests for shared Agent tool request validation, including required `query`, bounded `top_k`, bounded `max_retries`, and optional `source_types`.
- [x] 1.2 Create shared tool request/result models for `knowledge_base_search`, `collection_overview`, and `web_search`.
- [x] 1.3 Add conversion helpers that turn structured tool sources into LangChain `Document` objects and model-facing tool content.
- [x] 1.4 Add tests for structured result parsing from dictionaries, JSON strings, and plain text fallback values.

## 2. Knowledge-Base Search Tool Service

- [x] 2.1 Add failing tests for `knowledge_base_search` using `collection_name`, `source_types`, and `top_k` against a mocked vector store.
- [x] 2.2 Extract current ReAct Agent knowledge-base retrieval behavior into a shared service function.
- [x] 2.3 Apply `source_types` filtering by combining `chunk_level=child` with `source_type` constraints when provided.
- [x] 2.4 Preserve parent chunk enrichment, retrieval audit content, source metadata, and empty-result fallback behavior in the shared service.
- [x] 2.5 Add retry/failure containment so exhausted failures return structured `status=error` results instead of breaking Agent endpoints.

## 3. Web Search and Metadata Shared Services

- [x] 3.1 Add failing tests for web search structured result success, empty result, retry, and provider failure cases.
- [x] 3.2 Wrap existing `run_web_search` behavior in the shared Agent tool result schema.
- [x] 3.3 Add bounded retry handling around web search provider execution.
- [x] 3.4 Move collection overview behavior into the shared service layer with structured result output.
- [x] 3.5 Ensure empty web search results return `status=empty` and do not contribute supporting source documents.

## 4. React Agent Local Tool Integration

- [x] 4.1 Update ReAct Agent local `@tool` wrappers to call the shared service layer.
- [x] 4.2 Update non-streaming tool message handling to parse structured tool outputs and merge returned sources into the existing `documents` list.
- [x] 4.3 Update streaming tool event handling to parse structured tool outputs and merge returned sources into the existing `documents` list.
- [x] 4.4 Keep existing `tool_calls`, `trace`, `debug_events`, grounding guard, and `QueryResponse.sources` behavior compatible with current clients.

## 5. MCP Server and Client Mode

- [x] 5.1 Add MCP and LangChain MCP adapter dependencies.
- [x] 5.2 Add configuration for `MCP_ENABLED`, MCP server URL, optional tool name prefixing, timeout, and local fallback behavior.
- [x] 5.3 Implement an MCP server module exposing `knowledge_base_search`, `collection_overview`, and `web_search` through the shared service layer.
- [x] 5.4 Implement a React Agent MCP tool loader that lazily imports MCP dependencies and loads configured server tools when MCP mode is enabled.
- [x] 5.5 Add fallback behavior that records MCP loading failures and returns local tools when fallback is enabled.
- [x] 5.6 Add tests proving local mode remains default and MCP mode passes loaded tools to the ReAct Agent runtime.

## 6. Documentation and Verification

- [x] 6.1 Update `.env.example` with MCP configuration and comments.
- [x] 6.2 Update `README.md` to describe local tool mode, MCP tool server mode, structured tool results, and document type filtering.
- [x] 6.3 Run focused unit tests for shared tools, React Agent source handling, web search, and MCP configuration.
- [x] 6.4 Run the broader existing test suite and fix regressions.
