## Context

LangSmith tracing is implemented for React Agent (`create_agent` + `RunnableConfig`) and Plan-Execute Agent (root `trace()` + LangChain chains + manual step `trace()`). Code review found that LangChain chain runs and manual `trace()` blocks are not consistently parented under the per-request root run, so LangSmith UI shows parallel top-level runs. Execution-phase traces also omit the unified metadata schema (`route`, `endpoint`, `streaming`, `collection_name`) defined in the original design.

This change is a corrective pass on the archived change `add-langsmith-observability-for-react-and-plan-execute`. No API contract changes.

## Goals / Non-Goals

**Goals:**

- One identifiable root run per React or Plan-Execute API request in LangSmith.
- Child runs (planning chain, execution wrapper, step traces, synthesis chain) nested under that root when tracing is enabled.
- All traced Plan-Execute phases and steps include the full unified metadata schema for filtering.
- Startup logs a clear warning and does not enable `LANGCHAIN_TRACING_V2` when tracing is requested without an API key.
- Unknown plan tools do not emit misleading successful step traces.

**Non-Goals:**

- LangGraph `/query` tracing hierarchy.
- Offline evaluation datasets or CI scoring in LangSmith.
- Propagating `request_id` from FastAPI middleware (can follow later).

## Decisions

### Introduce a request-scoped trace context

Add a small `AgentTraceContext` (dataclass or TypedDict) in `app/core/tracing.py` holding:

- `route`, `endpoint`, `streaming`, `collection_name`, `history_count`
- Optional reference to the current LangSmith parent run (e.g. `RunTree` from root `trace()`)

Factory: `build_agent_trace_context(route=..., endpoint=..., ...)`.

Child runs and `build_run_config` accept an optional `trace_context` and:

1. Merge `build_agent_metadata(**context fields, phase=...)` into every manual `trace()` metadata dict.
2. For LangChain `.invoke` / `ainvoke`, enter `langsmith.run_helpers.tracing_context` with the parent run **or** attach parent callbacks from the root run so chain runs appear as children.

Alternative considered: rely only on `RunnableConfig.metadata` without parent linking. Rejected because metadata alone does not fix tree fragmentation.

### Plan-Execute: single root `trace()` owns the request

Refactor `run_plan_execute_agent_query` and `stream_plan_execute_agent_query` so:

1. Open root `trace(name="plan-execute-query"|"plan-execute-stream", metadata=full schema)`.
2. Pass `AgentTraceContext` into `_create_plan`, `_execute_plan_steps`, `_synthesize_answer`.
3. Planning/synthesis chains use `build_run_config(..., trace_context=ctx)` inside the root tracing context.
4. `plan-execute-execution` and `plan-execute-step-*` use `parent` from context or run inside the same `tracing_context`.

Stream path: keep async generator structure; root `trace()` must span the full generator lifetime (context remains open until the final event is yielded).

### React Agent: root trace + child agent run

Wrap `run_react_agent_query` and `stream_react_agent_query` bodies in a root `trace(name="react-agent-query"|"react-agent-stream", metadata=full schema)` and pass `trace_context` into `build_run_config` for `ainvoke` / `astream_events` so the agent runnable nests under the root.

Alternative considered: only metadata on agent config without root wrapper. Rejected for the same hierarchy reason as Plan-Execute.

### Startup credential guard

In `configure_langsmith_tracing()`:

- If `langsmith_tracing_enabled` is true and `langsmith_api_key` is empty: log warning, **do not** set `LANGCHAIN_TRACING_V2`, return early.
- If key present: set env vars as today.

`is_tracing_enabled()` returns true only when flag is on **and** API key is non-empty (effective tracing), so call sites do not attempt traces that cannot export.

Alternative considered: fail fast at startup with exception. Rejected to avoid breaking local dev when flag is accidentally left on.

### Configuration alias cleanup

Remove `LANGCHAIN_TRACING_V2` from the **read** aliases of `langsmith_tracing_enabled`. Keep reading `LANGSMITH_TRACING_ENABLED` only. Document that the application writes `LANGCHAIN_TRACING_V2` when enabling tracing.

Reduces ambiguity when shell has `LANGCHAIN_TRACING_V2=false` but app settings intend otherwise.

### Unknown plan tool step handling

In `_execute_plan_steps`, when a step tool is not in the allowed executable set (or normalized to unsupported):

- Do not open a success step trace; either skip tracing or end with `step_status=skipped`.
- Do not append a misleading `tool_calls` success entry (preserve existing control-flow: skip unsupported tools).

### Rename shadowed variable in `build_run_config`

Rename inner dict variable from `config` to `run_config` to avoid shadowing `app.core.config`.

## Risks / Trade-offs

- **[Risk] LangSmith/LangChain version differences in parent linking APIs** → Mitigation: use `tracing_context` documented in installed `langsmith`; add unit tests with mocks; manual smoke in LangSmith UI.
- **[Risk] Stream generators exiting early may close root trace prematurely** → Mitigation: root `trace()` wraps the full async generator function body; document consumer must drain or close generator.
- **[Trade-off] Effective tracing disabled without key even if flag is true** → Clearer than silent failure; documented in README.

## Migration Plan

1. Update tracing helpers and config aliases.
2. Refactor Plan-Execute and React services to use `AgentTraceContext`.
3. Extend tests and README.
4. Operators with `LANGSMITH_TRACING_ENABLED=true` must set `LANGSMITH_API_KEY` (no behavior change when already configured).

## Open Questions

None blocking; `request_id` propagation deferred.
