## Context

The project already has a shared `web_search` tool with a provider abstraction and a deterministic `mock` provider. ReAct Agent and Plan-Execute Agent both call the shared facade, and web results are converted into source-compatible document evidence. The next step is to add a live provider without changing either Agent's public API or response shape.

Tavily is a good fit because its Search API is optimized for LLM and Agent use cases and returns result fields that map naturally to the current internal contract: title, URL, content snippet, and optional relevance score.

## Goals / Non-Goals

**Goals:**

- Add `tavily` as a supported web search provider.
- Keep `mock` as the default provider for local development and tests.
- Configure Tavily through environment settings without hard-coding secrets.
- Map Tavily result items into `WebSearchResult` and existing source-compatible documents.
- Contain external API errors as failed `web_search` tool output.
- Test response mapping, missing configuration, and error handling without making live network calls.

**Non-Goals:**

- Do not add browser scraping, crawling, or Tavily Extract.
- Do not fetch full page content unless explicitly configured through Tavily search response options.
- Do not change Agent endpoint URLs or `QueryResponse`.
- Do not make Tavily the default provider.

## Decisions

### Use Direct HTTP Instead of Tavily SDK

Implement the provider using a simple HTTP POST to Tavily's Search API rather than adding a Tavily SDK dependency. The project already has a small provider protocol, and direct HTTP keeps the implementation easy to test with mocked transport.

Alternative considered: add the official Tavily SDK. That may be convenient later, but it introduces another dependency and makes transport-level tests less direct.

### Add Provider-Specific Settings

Add settings such as `web_search_api_key`, `web_search_timeout_seconds`, `web_search_tavily_search_depth`, `web_search_tavily_include_raw_content`, and `web_search_tavily_max_raw_content_chars`. The common settings `web_search_provider` and `web_search_top_k` continue to select the provider and result count.

Alternative considered: use only one generic API key setting. A generic key is enough for one provider, but provider-specific knobs such as search depth and raw content inclusion need explicit configuration.

### Preserve the Existing Result Contract

The Tavily provider should map each Tavily result into `WebSearchResult(title, url, snippet)`. If Tavily returns `content`, use it as the snippet. If raw content is enabled and available, it can be used only as a bounded fallback when `content` is absent.

Alternative considered: expand `WebSearchResult` with provider-specific fields such as score and raw content. That can wait until the product needs to display or filter by those fields.

### Fail Closed on Misconfiguration and Provider Errors

When `WEB_SEARCH_PROVIDER=tavily` but the API key is missing, or Tavily returns an invalid response or HTTP error, the tool should return a failed tool output rather than crashing the Agent endpoint.

Alternative considered: raise errors to the FastAPI layer. That would expose provider failures as endpoint failures, which is less useful for an Agent tool that can report failed tool calls and continue with a guarded answer.

## Risks / Trade-offs

- Tavily costs and rate limits can affect runtime behavior -> Keep mock as default and document Tavily as opt-in.
- Live search makes tests flaky if used directly -> Mock the HTTP transport and avoid live network tests in unit tests.
- Response shape may evolve -> Parse defensively and ignore unknown fields.
- Raw content can be large -> Default raw content inclusion to false and keep snippets bounded by existing preview settings.
- Missing API key is common during local setup -> Return a clear failed tool output that names the missing configuration.

## Migration Plan

This is additive. Existing mock-backed behavior remains unchanged unless `WEB_SEARCH_PROVIDER=tavily` is configured. Deployment requires adding the new settings to `.env`, installing any new HTTP dependency if chosen, and setting `WEB_SEARCH_API_KEY`. Rollback is to set `WEB_SEARCH_PROVIDER=mock` or remove the Tavily provider branch.

## Open Questions

- Should Tavily relevance score be stored in document metadata in the first implementation?
- Should domain include/exclude filters be added now or left for a later provider tuning change?
