## 1. Configuration

- [x] 1.1 Add Tavily-related settings for API key, request timeout, search depth, and raw-content inclusion.
- [x] 1.2 Update `.env.example` with commented Tavily configuration values while keeping `mock` as the default provider.

## 2. Provider Implementation

- [x] 2.1 Add tests for selecting `tavily` through the existing provider factory.
- [x] 2.2 Implement `TavilyWebSearchProvider` using direct HTTP POST to Tavily Search API.
- [x] 2.3 Map Tavily `results` items into `WebSearchResult(title, url, snippet)` values.
- [x] 2.4 Ensure Tavily requests honor configured top-k, timeout, search depth, and raw-content settings.

## 3. Failure Handling

- [x] 3.1 Add tests for missing Tavily API key returning failed tool output with no documents.
- [x] 3.2 Add tests for Tavily HTTP/request failures returning failed tool output with no documents.
- [x] 3.3 Add tests for invalid or empty Tavily responses returning no supporting documents.
- [x] 3.4 Keep existing Agent endpoint behavior intact by containing provider errors inside `run_web_search`.

## 4. Documentation and Verification

- [x] 4.1 Update README web search documentation to describe `WEB_SEARCH_PROVIDER=tavily` setup and behavior.
- [x] 4.2 Run focused web search tests and the full unittest suite.
- [x] 4.3 Validate the OpenSpec change before implementation completion.
