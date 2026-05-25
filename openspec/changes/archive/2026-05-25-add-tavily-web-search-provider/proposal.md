## Why

The project now has a mock-backed `web_search` tool, but it cannot retrieve real external information. Adding a Tavily provider turns the existing Agent web search extension point into a usable live-search capability while preserving the deterministic mock provider for tests and local development.

## What Changes

- Add `tavily` as a supported `WEB_SEARCH_PROVIDER` value.
- Add Tavily API configuration for API key, request timeout, search depth, and optional raw-content inclusion.
- Map Tavily search results into the existing `WebSearchResult(title, url, snippet)` contract.
- Contain Tavily request, authentication, provider, and response-shape errors as failed `web_search` tool output instead of breaking Agent endpoints.
- Keep `mock` as the default provider and test path.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `agent-web-search-tool`: The shared Agent web search tool gains a real Tavily provider in addition to the existing mock provider.

## Impact

- Affected code: `app/services/web_search.py`, `app/core/config.py`, `.env.example`, README documentation, and tests.
- APIs: no endpoint or response schema changes; existing Agent responses continue using `tool_calls`, `trace`, `debug_events`, and `sources`.
- Dependencies: may add an HTTP client dependency if the implementation uses one not already present.
- Systems: Tavily is an external network dependency only when `WEB_SEARCH_PROVIDER=tavily` is configured.
