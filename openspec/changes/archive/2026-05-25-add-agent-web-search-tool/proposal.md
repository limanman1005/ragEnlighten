## Why

The project currently grounds agent answers only in the indexed vector store, so agents cannot represent external or current information as an explicit, traceable tool result. Adding a web search tool gives both agent modes a stable extension point for external evidence while keeping the first implementation deterministic through a mock provider.

## What Changes

- Add a shared `web_search` tool capability for Agent workflows.
- Introduce a provider abstraction with a deterministic mock provider as the initial implementation.
- Allow the ReAct Agent to call `web_search` when a question needs external or current information beyond the indexed knowledge base.
- Allow the Plan-Execute Agent planning phase to select `web_search` and execute it in plan order.
- Include web search calls in existing `tool_calls`, `trace`, `debug_events`, and final response grounding behavior.
- Add configuration hooks for enabling web search, choosing a provider, and setting result count, without requiring a real external search API in this change.

## Capabilities

### New Capabilities

- `agent-web-search-tool`: Covers the shared web search tool contract, provider abstraction, mock provider behavior, and how web evidence is surfaced in agent responses.

### Modified Capabilities

- `plan-execute-agent-api`: The Plan-Execute Agent's allowed planning tools expand to include `web_search`, and the execution contract records web search tool calls in plan order.

## Impact

- Affected code: `app/services/react_agent.py`, `app/services/plan_execute_agent.py`, new shared web search service module, `app/core/config.py`, tests, and README/API documentation.
- APIs: existing chat endpoints remain unchanged; response payloads may include `web_search` entries in `tool_calls`, `trace`, `debug_events`, and `sources`.
- Dependencies: no real web search dependency is required for the initial mock provider.
- Systems: external network access remains optional and disabled unless a future real provider is configured.
