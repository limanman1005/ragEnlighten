## 1. Shared Web Search Service

- [x] 1.1 Add web search configuration defaults for enablement, provider selection, and top-k result count.
- [x] 1.2 Create a shared web search service module with a provider protocol, structured result model, and `search_web` facade.
- [x] 1.3 Implement a deterministic mock web search provider that returns title, URL, and snippet fields without external credentials.
- [x] 1.4 Add formatting helpers that convert web search results into tool output text and source-compatible document evidence.

## 2. ReAct Agent Integration

- [x] 2.1 Register `web_search` as a LangChain tool in the ReAct Agent runtime.
- [x] 2.2 Track web search evidence alongside knowledge-base documents so final responses can include web sources.
- [x] 2.3 Update the ReAct Agent prompt to use `web_search` for external or current information and preserve knowledge-base retrieval for indexed document questions.
- [x] 2.4 Ensure ReAct non-streaming and streaming debug events expose `web_search` tool starts, outputs, and final grounding status.

## 3. Plan-Execute Agent Integration

- [x] 3.1 Add `web_search` to the allowed Plan-Execute planning tools and normalize common web search aliases.
- [x] 3.2 Update the planning prompt so external or current information can produce a `web_search` step before answer synthesis.
- [x] 3.3 Extend plan execution to run `web_search` in plan order and record the call in `tool_calls`, `trace`, and `debug_events`.
- [x] 3.4 Treat successful web search output as supporting context for validation and grounding.

## 4. Tests

- [x] 4.1 Add unit tests for the mock provider and shared web search formatting.
- [x] 4.2 Add Plan-Execute tests for parsing, normalizing, and executing `web_search` steps in order.
- [x] 4.3 Add Agent result tests verifying web search evidence appears in tool calls and source-compatible output.
- [x] 4.4 Run the focused test suite and fix regressions.

## 5. Documentation

- [x] 5.1 Update `.env.example` with web search configuration fields and mock-provider defaults.
- [x] 5.2 Update README Agent documentation to describe the `web_search` tool, mock provider behavior, and future real provider extension point.
