## ADDED Requirements

### Requirement: Tavily web search provider

The system SHALL support `tavily` as a real web search provider for the shared Agent `web_search` tool.

#### Scenario: Tavily provider is selected

- **WHEN** web search is enabled and `WEB_SEARCH_PROVIDER` is set to `tavily`
- **THEN** `web_search` uses Tavily Search API behavior instead of the mock provider

#### Scenario: Tavily provider returns structured evidence

- **WHEN** Tavily returns search results for a query
- **THEN** the provider maps each result into structured evidence containing title, URL, and snippet fields

#### Scenario: Tavily provider respects result limit

- **WHEN** `web_search` is called with a configured top-k result limit
- **THEN** the Tavily request asks for no more than the configured number of results

### Requirement: Tavily provider configuration

The system SHALL configure Tavily without hard-coded credentials.

#### Scenario: Tavily API key is required

- **WHEN** `WEB_SEARCH_PROVIDER=tavily` and no Tavily API key is configured
- **THEN** `web_search` returns a failed tool output explaining the missing configuration

#### Scenario: Tavily request timeout is configurable

- **WHEN** a Tavily search request is made
- **THEN** the request uses the configured web search timeout value

#### Scenario: Tavily search options are configurable

- **WHEN** Tavily search depth or raw-content inclusion settings are configured
- **THEN** the Tavily request includes those configured search options

### Requirement: Tavily provider failure containment

The Tavily provider SHALL contain provider errors without breaking Agent endpoints.

#### Scenario: Tavily HTTP error is contained

- **WHEN** Tavily returns an HTTP error or the request fails
- **THEN** `web_search` records failed tool output and returns no supporting source documents

#### Scenario: Tavily invalid response is contained

- **WHEN** Tavily returns a response without usable search results
- **THEN** `web_search` returns no supporting source documents and does not treat the response as grounded evidence
