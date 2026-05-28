## Requirements

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

### Requirement: Plan-Execute execution step observability

The system SHOULD record Plan-Execute tool execution steps in LangSmith in a way that can be correlated with response `tool_calls`.

#### Scenario: Execution steps are traceable

- **WHEN** the Plan-Execute Agent executes one or more plan steps that invoke tools
- **THEN** LangSmith trace data includes per-step identifiers and tool names for each executed step
- **AND** step status (success or failure) is observable in the trace

#### Scenario: Execution steps align with tool_calls

- **WHEN** a Plan-Execute response includes `tool_calls` for executed plan steps
- **THEN** traced execution steps use tool names and step ordering consistent with the recorded `tool_calls`

### Requirement: Accurate Plan-Execute step trace status

The system SHALL NOT record successful Plan-Execute step traces for tools that were not executed.

#### Scenario: Unsupported plan tool is skipped

- **WHEN** a plan step references a tool that is not executed by the Plan-Execute runtime
- **THEN** LangSmith does not record that step as a successful tool run
- **AND** any recorded step status reflects `skipped` or the step trace is omitted

### Requirement: Backward compatibility for agent APIs

Enabling LangSmith tracing SHALL NOT change the external behavior of React or Plan-Execute agent endpoints.

#### Scenario: Response shape unchanged

- **WHEN** tracing is enabled and a client calls any React or Plan-Execute agent endpoint
- **THEN** the response remains `QueryResponse`-compatible for non-streaming calls
- **AND** SSE streaming event types and framing remain unchanged

#### Scenario: Agent semantics unchanged

- **WHEN** tracing is enabled
- **THEN** tool selection, plan generation, grounding guards, and validation behavior remain the same as when tracing is disabled

### Requirement: Evaluation outputs include trace correlation metadata

When LangSmith tracing is enabled, the system SHALL expose enough trace correlation metadata in evaluation outputs to support failure triage without changing agent API response schemas.

#### Scenario: Correlation fields are captured for traced runs

- **WHEN** an evaluation case executes while LangSmith tracing is effectively enabled
- **THEN** evaluation output includes route, endpoint, and phase metadata for the run
- **AND** includes available trace or run identifiers needed to locate the request in LangSmith

#### Scenario: Evaluation remains functional without LangSmith export

- **WHEN** LangSmith tracing is disabled or not exportable
- **THEN** evaluation outputs still include local case, mode, and response-derived analysis fields
- **AND** missing LangSmith identifiers do not fail the evaluation run
