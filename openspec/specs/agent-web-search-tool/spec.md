## Requirements

### Requirement: Shared Agent web search tool

The system SHALL provide a `web_search` tool that can be used by supported Agent workflows to retrieve external web search evidence.

#### Scenario: Web search tool returns structured evidence

- **WHEN** an Agent calls `web_search` with a non-empty query
- **THEN** the tool returns structured search results containing title, URL, and snippet fields

#### Scenario: Web search tool is recorded in Agent traces

- **WHEN** an Agent executes `web_search`
- **THEN** the response records the tool call in `tool_calls` and includes corresponding trace or debug information

### Requirement: Mock web search provider

The system SHALL include a deterministic mock web search provider that does not require external network access or API credentials.

#### Scenario: Mock provider works without external credentials

- **WHEN** web search is configured to use the mock provider
- **THEN** `web_search` returns deterministic mock results without requiring a provider API key

#### Scenario: Mock provider respects result limit

- **WHEN** `web_search` is called with a configured top-k result limit
- **THEN** the mock provider returns no more than the configured number of results

### Requirement: ReAct Agent web search access

The ReAct Agent SHALL have access to the `web_search` tool for questions that need external or current information beyond the indexed knowledge base.

#### Scenario: ReAct Agent can call web search

- **WHEN** a client sends a question to the ReAct Agent endpoint that requires external web information
- **THEN** the Agent can execute `web_search` and include the resulting evidence in the final response context

#### Scenario: ReAct Agent preserves existing retrieval behavior

- **WHEN** a client sends a question that can be answered from indexed knowledge-base content
- **THEN** the ReAct Agent remains able to use `knowledge_base_search` and existing endpoint behavior remains available

### Requirement: Web search evidence in QueryResponse-compatible output

Agent web search evidence SHALL be exposed through the existing `QueryResponse`-compatible payload without requiring a new public response schema.

#### Scenario: Web results appear as sources

- **WHEN** an Agent uses web search results as supporting evidence
- **THEN** the final response includes source entries derived from those web results where each source identifies the result URL and content snippet

#### Scenario: Web search supports grounding validation

- **WHEN** an Agent answer is based on successful web search output
- **THEN** validation treats the web search output as supporting context and does not fail solely because no vector-store chunks were retrieved

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

### Requirement: Web search uses shared Agent tool schema

The `web_search` tool SHALL support the shared Agent tool result shape used by local and MCP Agent tools.

#### Scenario: Web search returns structured tool result

- **WHEN** `web_search` completes successfully
- **THEN** the result contains `status`, `content`, `sources`, and no required new public API response schema

#### Scenario: Web search sources preserve URL evidence

- **WHEN** `web_search` returns source entries
- **THEN** each source entry identifies the result URL, title, snippet or content preview, and `source_type=web`

### Requirement: Web search retry and failure containment

The `web_search` tool SHALL contain provider failures and support bounded retry behavior at the shared Agent tool layer.

#### Scenario: Web search retries transient failure

- **WHEN** `web_search` receives a transient provider failure and retries remain
- **THEN** the tool retries the provider call up to the validated retry limit before returning a failed result

#### Scenario: Web search failure returns structured error

- **WHEN** `web_search` exhausts retries or encounters a non-recoverable provider error
- **THEN** the tool returns a structured result with `status=error`, no supporting sources, and an error message suitable for `tool_calls`

#### Scenario: Web search empty result is not grounding evidence

- **WHEN** `web_search` returns no usable search results
- **THEN** the tool returns `status=empty`, empty `sources`, and the Agent does not treat the result as grounded evidence
