## Requirements

### Requirement: MCP server exposes Agent tools

The system SHALL provide an MCP server that exposes Agent retrieval and metadata tools using the same business logic as the local Agent tool path.

#### Scenario: MCP server lists Agent tools

- **WHEN** an MCP client connects to the Agent tool server
- **THEN** the server exposes `knowledge_base_search`, `collection_overview`, and `web_search` tools

#### Scenario: MCP tools use shared tool behavior

- **WHEN** an MCP client calls an Agent tool
- **THEN** the tool uses the same shared service behavior as the local ReAct Agent tool with equivalent input values

### Requirement: Structured knowledge-base search input

The `knowledge_base_search` tool SHALL accept validated structured input for query text, optional knowledge-base collection, optional document type filters, result limit, timeout, and retry count.

#### Scenario: Query is required

- **WHEN** `knowledge_base_search` is called without a non-empty `query`
- **THEN** the tool returns a failed structured result explaining the missing required query

#### Scenario: Collection selects knowledge base

- **WHEN** `knowledge_base_search` is called with `collection_name`
- **THEN** retrieval uses that collection instead of the default collection

#### Scenario: Document type filter limits retrieval

- **WHEN** `knowledge_base_search` is called with `source_types`
- **THEN** retrieval limits returned child chunks to matching `source_type` metadata values

#### Scenario: Result limit is bounded

- **WHEN** `knowledge_base_search` is called with `top_k`
- **THEN** retrieval requests no more than the validated bounded result limit

### Requirement: Structured Agent tool results

Agent tools exposed locally and through MCP SHALL return a structured result containing status, model-facing content, source metadata, and optional error details.

#### Scenario: Successful retrieval includes sources

- **WHEN** `knowledge_base_search` finds matching chunks
- **THEN** the result status is `success`, `content` summarizes the retrieved evidence, and `sources` contains source-compatible metadata for each returned chunk

#### Scenario: Empty retrieval is explicit

- **WHEN** `knowledge_base_search` finds no matching chunks
- **THEN** the result status is `empty`, `content` states that no relevant documents were found, and `sources` is empty

#### Scenario: Failed tool call is contained

- **WHEN** an Agent tool fails after retries are exhausted
- **THEN** the result status is `error`, `error` explains the failure, and the Agent endpoint remains available

### Requirement: React Agent supports MCP tool mode

The ReAct Agent SHALL support a configuration-controlled MCP client mode while preserving local in-process tools as the default behavior.

#### Scenario: Local tools remain default

- **WHEN** MCP mode is disabled
- **THEN** the ReAct Agent uses local in-process tools and keeps existing endpoint behavior

#### Scenario: MCP mode loads tools

- **WHEN** MCP mode is enabled and the configured MCP server is reachable
- **THEN** the ReAct Agent loads MCP tools and passes them to the Agent runtime

#### Scenario: MCP loading falls back safely

- **WHEN** MCP mode is enabled but MCP tool loading fails
- **THEN** the ReAct Agent records the loading failure and falls back to local tools

### Requirement: MCP tool sources support grounding

The ReAct Agent SHALL convert structured MCP tool source metadata into the same internal document representation used by local tools.

#### Scenario: MCP retrieval sources appear in response

- **WHEN** the ReAct Agent receives MCP tool output with `sources`
- **THEN** the final `QueryResponse.sources` includes those sources using the existing response schema

#### Scenario: MCP retrieval sources count as grounding evidence

- **WHEN** the ReAct Agent answer is based on MCP tool sources
- **THEN** validation treats those sources as supporting evidence and does not fail solely because the local tool path was not used
