## ADDED Requirements

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
