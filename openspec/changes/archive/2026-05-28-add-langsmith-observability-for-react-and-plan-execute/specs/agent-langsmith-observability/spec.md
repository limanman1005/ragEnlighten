## ADDED Requirements

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

### Requirement: Unified trace metadata for agent runs

When LangSmith tracing is enabled, agent runs SHALL include consistent metadata for filtering and comparison across React and Plan-Execute flows.

#### Scenario: Metadata includes route and endpoint

- **WHEN** a React or Plan-Execute agent request is traced
- **THEN** the trace run metadata includes `route` (`react_agent` or `plan_execute_agent`)
- **AND** includes `endpoint` identifying the API path used for the request

#### Scenario: Metadata includes collection and streaming mode

- **WHEN** a traced agent request specifies a collection and uses a streaming or non-streaming endpoint
- **THEN** the trace run metadata includes `collection_name` and `streaming` (`true` or `false`)

### Requirement: React Agent LangSmith trace coverage

The system SHALL produce LangSmith traces for React Agent non-streaming and streaming invocations that cover the LangChain agent reasoning and tool-call chain.

#### Scenario: React non-streaming trace

- **WHEN** a client calls `POST /api/v1/chat/react-agent` with tracing enabled
- **THEN** LangSmith records a trace run named for React non-streaming queries
- **AND** the run metadata identifies `route=react_agent` and `streaming=false`

#### Scenario: React streaming trace

- **WHEN** a client calls `POST /api/v1/chat/react-agent/stream` with tracing enabled
- **THEN** LangSmith records a trace run named for React streaming queries
- **AND** the run metadata identifies `route=react_agent` and `streaming=true`

### Requirement: Plan-Execute Agent phase visibility in traces

The system SHALL produce LangSmith traces for Plan-Execute Agent requests that expose at least the planning and synthesis LLM phases.

#### Scenario: Plan-Execute non-streaming phase traces

- **WHEN** a client calls `POST /api/v1/chat/plan-execute-agent` with tracing enabled
- **THEN** LangSmith records trace activity for the planning phase before tool execution
- **AND** records trace activity for the synthesis phase after tool execution completes

#### Scenario: Plan-Execute streaming phase traces

- **WHEN** a client calls `POST /api/v1/chat/plan-execute-agent/stream` with tracing enabled
- **THEN** LangSmith records trace runs with metadata identifying `route=plan_execute_agent` and `streaming=true`
- **AND** planning and synthesis phases remain identifiable in the trace

### Requirement: Plan-Execute execution step observability

The system SHOULD record Plan-Execute tool execution steps in LangSmith in a way that can be correlated with response `tool_calls`.

#### Scenario: Execution steps are traceable

- **WHEN** the Plan-Execute Agent executes one or more plan steps that invoke tools
- **THEN** LangSmith trace data includes per-step identifiers and tool names for each executed step
- **AND** step status (success or failure) is observable in the trace

#### Scenario: Execution steps align with tool_calls

- **WHEN** a Plan-Execute response includes `tool_calls` for executed plan steps
- **THEN** traced execution steps use tool names and step ordering consistent with the recorded `tool_calls`

### Requirement: Backward compatibility for agent APIs

Enabling LangSmith tracing SHALL NOT change the external behavior of React or Plan-Execute agent endpoints.

#### Scenario: Response shape unchanged

- **WHEN** tracing is enabled and a client calls any React or Plan-Execute agent endpoint
- **THEN** the response remains `QueryResponse`-compatible for non-streaming calls
- **AND** SSE streaming event types and framing remain unchanged

#### Scenario: Agent semantics unchanged

- **WHEN** tracing is enabled
- **THEN** tool selection, plan generation, grounding guards, and validation behavior remain the same as when tracing is disabled
