## MODIFIED Requirements

### Requirement: LangSmith tracing configuration

The system SHALL support enabling LangSmith tracing through application configuration without changing agent API contracts.

#### Scenario: Tracing disabled by default

- **WHEN** the service starts without LangSmith tracing enabled
- **THEN** agent endpoints continue to answer questions with unchanged request and response shapes
- **AND** no LangSmith tracing environment variables are forced on by the application

#### Scenario: Tracing enabled with valid credentials

- **WHEN** LangSmith tracing is enabled and a valid API key and project are configured
- **THEN** the application configures LangChain-compatible tracing environment variables at startup
- **AND** subsequent LangChain and LangGraph runnable invocations may emit traces to the configured LangSmith project

#### Scenario: Tracing enabled without API key

- **WHEN** LangSmith tracing is enabled but no API key is configured
- **THEN** the application logs a warning at startup
- **AND** does not set `LANGCHAIN_TRACING_V2` to enable export
- **AND** agent endpoints continue to answer questions with unchanged request and response shapes

### Requirement: Unified trace metadata for agent runs

When LangSmith tracing is effectively enabled, agent runs SHALL include consistent metadata for filtering and comparison across React and Plan-Execute flows.

#### Scenario: Metadata includes route and endpoint

- **WHEN** a React or Plan-Execute agent request is traced
- **THEN** the trace run metadata includes `route` (`react_agent` or `plan_execute_agent`)
- **AND** includes `endpoint` identifying the API path used for the request

#### Scenario: Metadata includes collection and streaming mode

- **WHEN** a traced agent request specifies a collection and uses a streaming or non-streaming endpoint
- **THEN** the trace run metadata includes `collection_name` and `streaming` (`true` or `false`)

#### Scenario: Execution steps include full metadata

- **WHEN** Plan-Execute execution or per-step traces are recorded
- **THEN** each such trace includes the same `route`, `endpoint`, `streaming`, and `collection_name` metadata as the root request trace
- **AND** includes `phase=execution` and step fields (`step_id`, `tool_name`) where applicable

### Requirement: Plan-Execute Agent phase visibility in traces

The system SHALL produce LangSmith traces for Plan-Execute Agent requests that expose planning, execution, and synthesis as nested runs under a single root trace per API request.

#### Scenario: Plan-Execute non-streaming hierarchical trace

- **WHEN** a client calls `POST /api/v1/chat/plan-execute-agent` with effective tracing enabled
- **THEN** LangSmith records a root run for the request
- **AND** planning, execution, and synthesis activity appear as descendant runs of that root

#### Scenario: Plan-Execute streaming hierarchical trace

- **WHEN** a client calls `POST /api/v1/chat/plan-execute-agent/stream` with effective tracing enabled
- **THEN** LangSmith records a root run for the streaming request
- **AND** planning, execution, and synthesis activity appear as descendant runs of that root

## ADDED Requirements

### Requirement: React Agent hierarchical traces

The system SHALL nest React Agent LangChain activity under a single root trace per API request when effective tracing is enabled.

#### Scenario: React non-streaming hierarchical trace

- **WHEN** a client calls `POST /api/v1/chat/react-agent` with effective tracing enabled
- **THEN** LangSmith records a root run for the request
- **AND** the LangChain agent run appears as a descendant of that root

#### Scenario: React streaming hierarchical trace

- **WHEN** a client calls `POST /api/v1/chat/react-agent/stream` with effective tracing enabled
- **THEN** LangSmith records a root run for the streaming request
- **AND** the LangChain agent run appears as a descendant of that root

### Requirement: Accurate Plan-Execute step trace status

The system SHALL NOT record successful Plan-Execute step traces for tools that were not executed.

#### Scenario: Unsupported plan tool is skipped

- **WHEN** a plan step references a tool that is not executed by the Plan-Execute runtime
- **THEN** LangSmith does not record that step as a successful tool run
- **AND** any recorded step status reflects `skipped` or the step trace is omitted
