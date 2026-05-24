## Requirements

### Requirement: Plan-execute Agent non-streaming endpoint

The system SHALL expose a non-streaming plan-execute Agent endpoint that accepts the same question, collection, and history inputs as the existing React Agent chat endpoint and returns a `QueryResponse`-compatible payload.

#### Scenario: Successful non-streaming plan-execute answer

- **WHEN** a client sends `POST /api/v1/chat/plan-execute-agent` with a valid question
- **THEN** the system returns a response containing `answer`, `plan`, `tool_calls`, `sources`, `trace`, `confidence_score`, and `validation`

#### Scenario: Existing endpoints remain unchanged

- **WHEN** the plan-execute Agent endpoint is added
- **THEN** `/api/v1/query`, `/api/v1/chat/react-agent`, and `/api/v1/chat/react-agent/stream` remain available with their existing route behavior

### Requirement: Explicit planning before execution

The system SHALL complete an explicit planning phase before performing retrieval, metadata lookup, or final answer synthesis for the plan-execute Agent.

#### Scenario: Plan is returned before execution details

- **WHEN** a client calls the plan-execute Agent
- **THEN** the response `plan` contains ordered human-readable steps created before the recorded execution `tool_calls`

#### Scenario: Execution follows the generated plan

- **WHEN** the planning phase produces retrieval or metadata steps
- **THEN** the system executes those steps in plan order and records each executed tool call in `tool_calls`

### Requirement: Plan-execute Agent SSE endpoint

The system SHALL expose an SSE streaming endpoint for the plan-execute Agent using `text/event-stream`.

#### Scenario: Streaming response uses SSE framing

- **WHEN** a client sends `POST /api/v1/chat/plan-execute-agent/stream`
- **THEN** the response media type is `text/event-stream` and events are emitted using `event:` and `data:` fields

#### Scenario: Streaming emits plan before execution events

- **WHEN** the plan-execute Agent stream starts processing a valid question
- **THEN** the stream emits a `plan` event before execution `trace`, `debug`, `token`, or `final` events that depend on executing the plan

#### Scenario: Streaming emits final response

- **WHEN** the plan-execute Agent stream completes successfully
- **THEN** it emits a `final` event containing a `QueryResponse`-compatible payload

### Requirement: Grounding and validation

The plan-execute Agent SHALL ground document questions in retrieved vector-store evidence and report validation status.

#### Scenario: Grounded answer includes sources

- **WHEN** the plan-execute Agent retrieves relevant knowledge-base chunks
- **THEN** the final response includes those chunks in `sources` and reports validation with `citations_verified=true`

#### Scenario: Ungrounded answer is guarded

- **WHEN** the plan-execute Agent cannot retrieve grounded evidence for a document question
- **THEN** the final answer does not present unsupported factual claims and the validation issues explain the missing evidence
