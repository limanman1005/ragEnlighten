## Context

The application currently exposes two question-answering styles: `/api/v1/query` runs a LangGraph RAG workflow, and `/api/v1/chat/react-agent` plus `/api/v1/chat/react-agent/stream` runs a LangChain ReAct Agent that can decide to call tools while reasoning. The ReAct Agent is flexible, but its plan is implicit in the sequence of thoughts and tool calls. Users now need a separate Agent interface whose execution is easier to audit: the model must first produce a concrete plan, then execute that plan.

The implementation should reuse the existing request and response shape where possible. `ReactAgentQueryRequest` already carries `question`, `collection_name`, and `history`; `QueryResponse` already carries `answer`, `plan`, `tool_calls`, `sources`, `trace`, `debug_events`, `confidence_score`, and `validation`. Existing SSE formatting is available through `app.api.sse.format_sse_event`.

## Goals / Non-Goals

**Goals:**

- Add non-streaming and SSE streaming endpoints for a plan-first Agent.
- Make the planning phase explicit and observable in both response `plan` and stream events.
- Execute only after planning is complete.
- Reuse existing knowledge-base retrieval, collection metadata, source formatting, grounding guard, and validation conventions.
- Keep current LangGraph and ReAct endpoints unchanged.

**Non-Goals:**

- Do not replace the existing ReAct Agent endpoints.
- Do not change `/api/v1/query` LangGraph behavior.
- Do not introduce a new vector store, embedding provider, or LLM provider.
- Do not implement long-running background jobs or persisted agent sessions.

## Decisions

### Add a separate plan-execute service module

Create `app/services/plan_execute_agent.py` instead of folding the behavior into `react_agent.py`. The existing file is already responsible for ReAct execution, DeepSeek reasoning preservation, streaming event parsing, grounding, and result construction. A separate module keeps the plan-first contract clear and avoids mixing two different agent styles in one service.

Alternative considered: add flags to `run_react_agent_query`. This would reduce file count but would blur the contract and risk accidentally preserving ReAct behavior where the model can keep deciding tool calls during execution.

### Use a two-phase workflow

The new service should run:

1. Planning phase: use the LLM to create a structured JSON plan with ordered steps, each step containing a step id, purpose, tool name, and query or instruction.
2. Execution phase: iterate over the completed plan in order, call the allowed tools deterministically, collect documents/tool outputs, then synthesize the final answer from the execution results.

This is intentionally different from ReAct. During execution, the model should not decide new tool calls. If the plan is invalid or empty, the service should fail safely with a guarded answer and validation issue.

Alternative considered: use LangGraph for the plan-execute Agent. LangGraph would work, but this change is primarily a new Agent API and can be simpler as a service-level two-phase runner. LangGraph can be introduced later if planning/execution needs branching, retries, or persistence.

### Reuse the existing response model

The endpoints should return `QueryResponse` through the existing `_build_query_response` helper. The result dictionary should use `route="plan_execute_agent"` and include the explicit plan in the `plan` field. This preserves frontend compatibility with existing agent detail panels.

Alternative considered: create dedicated request/response schemas. That would make the contract more explicit, but it would duplicate fields already used by both LangGraph and ReAct modes.

### Support SSE from the start

Add `POST /api/v1/chat/plan-execute-agent/stream` with `text/event-stream`. Stream event types should include `plan`, `trace`, `debug`, `token`, `final`, and `error`. Planning must complete before execution events are emitted. Non-streaming endpoint should be `POST /api/v1/chat/plan-execute-agent`.

Alternative considered: add only non-streaming first. The UI already supports SSE for React Agent, so adding streaming now keeps the new API aligned with current frontend behavior.

## Risks / Trade-offs

- Plan quality risk -> Validate plan structure before execution and fall back to a minimal retrieval plan when the model returns malformed JSON.
- Reduced adaptability -> This is intentional; callers choose this endpoint when auditability matters more than dynamic ReAct behavior.
- Duplicate retrieval helper logic -> Start by reusing or extracting small helper functions from `react_agent.py` only where necessary; avoid a broad refactor in the first implementation.
- Streaming token grounding risk -> Follow the existing guarded behavior: suppress or replace ungrounded final answers when no documents were collected.

## Migration Plan

1. Add the new service and endpoints without changing existing routes.
2. Add tests around plan generation/execution ordering and SSE event formatting.
3. Optionally add a Streamlit chat mode after the backend contract is stable.
4. Rollback is deleting the new endpoints/service because no existing behavior is modified.
