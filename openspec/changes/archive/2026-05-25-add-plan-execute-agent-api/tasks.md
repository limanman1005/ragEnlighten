## 1. Tests and Service Skeleton

- [x] 1.1 Add unit tests for plan parsing and fallback behavior in `tests/test_plan_execute_agent.py`.
- [x] 1.2 Add unit tests that verify execution does not start until a plan is produced.
- [x] 1.3 Create `app/services/plan_execute_agent.py` with public functions `run_plan_execute_agent_query` and `stream_plan_execute_agent_query`.
- [x] 1.4 Define internal plan step data structures and validation helpers for ordered tool execution.

## 2. Plan-First Agent Execution

- [x] 2.1 Implement the planning phase using the configured `ChatOpenAI` model and a JSON-only planning prompt.
- [x] 2.2 Implement deterministic execution of planned `knowledge_base_search` and `collection_overview` steps.
- [x] 2.3 Reuse Chroma child chunk retrieval and parent chunk enrichment behavior from the existing React Agent service.
- [x] 2.4 Implement answer synthesis from completed plan outputs without allowing new tool decisions during execution.
- [x] 2.5 Add grounding guard, confidence score, validation report, trace, debug events, and `route="plan_execute_agent"` result construction.

## 3. FastAPI Endpoints

- [x] 3.1 Add `POST /api/v1/chat/plan-execute-agent` using `ReactAgentQueryRequest` and returning `QueryResponse`.
- [x] 3.2 Add `POST /api/v1/chat/plan-execute-agent/stream` returning `text/event-stream`.
- [x] 3.3 Emit SSE events in order: planning start, `plan`, execution `trace` or `debug`, optional `token`, and `final`.
- [x] 3.4 Convert final stream payloads through the existing `_build_query_response` helper before emitting the `final` event.
- [x] 3.5 Return SSE `error` events for stream failures without changing existing React Agent stream behavior.

## 4. UI and Documentation

- [x] 4.1 Add a Streamlit chat mode for the plan-execute Agent if the backend endpoint contract is stable.
- [x] 4.2 Update README endpoint documentation with non-streaming and SSE plan-execute Agent examples.
- [x] 4.3 Document the semantic difference between ReAct Agent and plan-execute Agent.

## 5. Verification

- [x] 5.1 Run the new plan-execute Agent unit tests and confirm they pass.
- [x] 5.2 Run existing SSE tests and confirm the React Agent stream format remains valid.
- [x] 5.3 Run linter diagnostics for modified Python files and fix introduced issues.
- [x] 5.4 Manually smoke test the non-streaming endpoint with a document question.
- [x] 5.5 Manually smoke test the streaming endpoint and confirm the first execution-dependent event occurs after the `plan` event.
