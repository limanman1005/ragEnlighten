## Context

The current project has three answer paths: the LangGraph RAG workflow, a ReAct Agent, and a Plan-Execute Agent. The user-facing chat modes already expose tool calls, traces, debug events, sources, confidence, and validation through a shared `QueryResponse` shape. The ReAct Agent defines LangChain tools inline, while the Plan-Execute Agent uses an explicit allowlist and a local executor.

The requested change targets the two Agent modes. Both need access to external web evidence, but the first implementation should not depend on a real search API key, network availability, or provider-specific response formats.

## Goals / Non-Goals

**Goals:**

- Add a `web_search` tool that both Agent modes can use.
- Keep the initial provider deterministic by implementing a mock provider first.
- Preserve existing endpoint URLs and response schemas.
- Record web search usage through existing `tool_calls`, `trace`, and `debug_events`.
- Represent web search evidence in a way that can be rendered through existing source display logic.
- Leave a clear extension point for later real providers such as Tavily, Serper, Brave, or Bing.

**Non-Goals:**

- Do not implement a live external search provider in this change.
- Do not add browser automation, scraping, crawling, or page content extraction.
- Do not modify the baseline `/query` LangGraph workflow.
- Do not introduce a new public API endpoint for search.

## Decisions

### Use a Shared Provider Abstraction

Create a small shared web search service, for example `app/services/web_search.py`, with a provider protocol and a `search_web(query, top_k)` facade. Both Agent implementations should call this shared facade rather than duplicating provider logic.

Alternative considered: define the mock search function separately in each agent module. That is faster initially, but it would make later real provider integration harder and risks inconsistent formatting between ReAct and Plan-Execute.

### Start With a Mock Provider

The default provider should be deterministic and local. It should return predictable result objects containing at least `title`, `url`, and `snippet`, capped by a configurable top-k value.

Alternative considered: integrate a live free provider immediately. That would make demos feel more realistic, but it would add network and rate-limit variability before the tool contract and tests are stable.

### Surface Web Results as Source-Compatible Evidence

Web search results should be converted into source-compatible document-like evidence with metadata such as `source`, `title`, `source_type`, and `retrieval_hop`. This lets the existing response builder include web results in `sources` without changing the API schema.

Alternative considered: only expose web results in `tool_calls`. That would preserve evidence in debug output, but users would not see web evidence alongside knowledge-base sources.

### Keep Grounding Strict but Broaden Supporting Context

For Agent modes, successful web search output should count as supporting context for validation and grounding. Answers should still be instructed to use only completed tool outputs and to say when tool evidence is insufficient.

Alternative considered: require vector-store documents for validation even after web search. That would make web search useless for questions that intentionally ask about external or current information.

### Gate Real Providers Behind Configuration

Add configuration fields such as `web_search_enabled`, `web_search_provider`, and `web_search_top_k`. The mock provider can work without an API key. Future real providers can add provider-specific keys without changing the Agent tool contract.

Alternative considered: always enable web search. Default-on external tools can surprise local users and make tests less predictable, so provider behavior should be explicit and configurable.

## Risks / Trade-offs

- Mock results may look artificial -> Keep titles, URLs, and snippets clearly deterministic and document that the first provider is a test/demo provider.
- The model may overuse web search -> Prompts should say to use `web_search` for external/current information or when indexed evidence is insufficient, not for every document question.
- Existing source schema has vector-store-oriented fields -> Use existing optional fields and metadata without expanding the public schema unless implementation proves it necessary.
- Plan generation may produce unsupported tool aliases -> Extend tool normalization so common terms like `web`, `web_search`, and `internet_search` map to `web_search`.
- Future live providers may fail or time out -> Keep provider errors contained in the tool output and recorded as failed tool calls without breaking the endpoint.

## Migration Plan

This change is additive. Existing endpoints remain unchanged. Deploy by adding the shared web search module, registering the tool in both Agent modes, adding configuration defaults, and updating tests/docs. Rollback is to remove `web_search` from the Agent tool lists and Plan-Execute allowlist while leaving existing RAG behavior untouched.

## Open Questions

- Which live provider should be implemented after the mock contract is stable?
- Should future web search results include fetched page content, or only search result snippets?
