## Context

The application exposes three question-answering modes: LangGraph RAG (`/query`), ReAct Agent (`/chat/react-agent` and `/stream`), and Plan-Execute Agent (`/chat/plan-execute-agent` and `/stream`). React Agent uses LangChain `create_agent` with `ainvoke` and `astream_events`. Plan-Execute Agent uses separate LangChain chains for planning and synthesis plus a Python loop (`_execute_plan_steps`) for deterministic tool execution. Neither path currently sends traces to LangSmith. Application-level `trace`, `debug_events`, and `tool_calls` remain useful for API consumers but do not provide cross-request LLM/tool observability or filtering in LangSmith.

This change is scoped to React and Plan-Execute agent APIs only. LangGraph `/query` tracing may reuse the same helper later but is out of scope for the initial implementation.

## Goals / Non-Goals

**Goals:**

- Enable LangSmith tracing for React and Plan-Execute agent endpoints (non-stream and stream).
- Define a single metadata schema and run naming convention so runs are filterable by `route`, `endpoint`, `collection_name`, and `streaming`.
- Ensure Plan-Execute traces show at least planning and synthesis LLM phases; recommend execution-step visibility mappable to `tool_calls`.
- Keep tracing off by default; enable via configuration without code changes.
- Preserve existing API contracts (`QueryResponse`, SSE events, agent behavior).

**Non-Goals:**

- Offline evaluation datasets, evaluators, or CI regression scoring in LangSmith.
- Refactoring Plan-Execute into LangGraph or merging with ReAct Agent.
- Tracing document ingestion, vector store operations, or `/query` LangGraph workflow in the first release.
- Changing answer quality, tool selection logic, or validation heuristics.

## Decisions

### Add `app/core/tracing.py` for bootstrap and run config

Create a dedicated module that:

1. Applies LangChain-compatible environment variables at startup when tracing is enabled (`LANGCHAIN_TRACING_V2`, `LANGCHAIN_API_KEY`, `LANGCHAIN_PROJECT`, optional `LANGCHAIN_ENDPOINT`).
2. Exposes `build_run_config(run_name, ...)` returning a dict suitable for `ainvoke`, `astream_events`, and LangChain `.invoke` calls.
3. Exposes `merge_run_configs` to combine tracing config with LangGraph options such as `recursion_limit`.

Alternative considered: set only raw environment variables and rely on defaults. Rejected because React and Plan-Execute need consistent `run_name`, `tags`, and `metadata` for filtering.

### Configuration via `Settings` with LangSmith and LangChain aliases

Add to `app/core/config.py`:

- `langsmith_tracing_enabled` (aliases: `LANGSMITH_TRACING_ENABLED`, `LANGCHAIN_TRACING_V2`)
- `langsmith_api_key` (aliases: `LANGSMITH_API_KEY`, `LANGCHAIN_API_KEY`)
- `langsmith_project` (default `ragEnlighten`, aliases: `LANGSMITH_PROJECT`, `LANGCHAIN_PROJECT`)
- `langsmith_endpoint` (optional, aliases: `LANGSMITH_ENDPOINT`, `LANGCHAIN_ENDPOINT`)

Call `configure_langsmith_tracing()` from `app/main.py` before serving requests.

Alternative considered: require operators to set `LANGCHAIN_*` only in `.env` without app settings. Rejected to keep configuration documented alongside other service settings in `.env.example`.

### Unified metadata schema

All agent runs MUST include metadata fields when tracing is enabled:

| Field | Description |
|-------|-------------|
| `route` | `react_agent` or `plan_execute_agent` |
| `endpoint` | API path identifier (e.g. `/api/v1/chat/react-agent`) |
| `streaming` | `true` or `false` |
| `collection_name` | Resolved collection for the request |
| `history_count` | Number of history messages |
| `phase` | Plan-Execute only: `planning`, `execution`, `synthesis` where applicable |

Optional fields: `request_id` (if propagated from API layer in a follow-up), `step_id`, `tool_name`, `step_status` for execution steps.

`tags` MUST include the primary `run_name` for quick filtering.

### Run naming convention

| Flow | `run_name` |
|------|------------|
| React non-stream | `react-agent-query` |
| React stream | `react-agent-stream` |
| Plan-Execute non-stream (root) | `plan-execute-query` |
| Plan-Execute stream (root) | `plan-execute-stream` |
| Plan-Execute planning chain | `plan-execute-planning` |
| Plan-Execute synthesis chain | `plan-execute-synthesis` |
| Plan-Execute execution wrapper | `plan-execute-execution` |

### React Agent instrumentation

Pass merged `RunnableConfig` to:

- `agent.ainvoke(model_input, config=...)`
- `agent.astream_events(..., config=...)`

Metadata: `route=react_agent`, appropriate `endpoint` and `streaming`.

Alternative considered: wrap the entire service in a LangSmith `@traceable` root only. Rejected as insufficient for tool-level detail; LangChain agent tracing is richer when config is passed to the agent runnable.

### Plan-Execute Agent instrumentation (two tiers)

**Tier 1 (required):**

- Pass config to planning chain `.invoke` with `phase=planning`.
- Pass config to synthesis chain `.invoke` with `phase=synthesis`.
- Wrap `run_plan_execute_agent_query` / stream entry with root run metadata (`plan-execute-query` or `plan-execute-stream`).

**Tier 2 (recommended in same change):**

- Wrap `_execute_plan_steps` with LangSmith `@traceable` (or equivalent child run) named `plan-execute-execution`.
- Record per-step metadata: `step_id`, `tool_name`, `step_status` (`success` / `failed`), `step_latency_ms`, optional `docs_found_count`.
- Ensure step records align with entries later appended to `tool_calls` for UI/trace correlation.

Alternative considered: rely only on planning/synthesis LLM traces. Rejected for Plan-Execute because the execution gap is the main observability blind spot discussed in exploration.

### Tracing disabled behavior

When `langsmith_tracing_enabled` is false, `build_run_config` returns only non-tracing keys (e.g. `recursion_limit` if passed). No LangSmith env vars are forced on. Existing log-based debugging unchanged.

### Dependency

Add `langsmith>=0.1.0` to `requirements.txt`. LangChain tracing uses the LangSmith client when env vars are set.

## Risks / Trade-offs

- **[Risk] Plan-Execute execution still partially invisible if Tier 2 is deferred** → Mitigation: spec requires at least planning/synthesis; tasks list Tier 2 as recommended same-change item.
- **[Risk] Sensitive data in traces (questions, retrieved content)** → Mitigation: document that operators should use LangSmith project access controls; optional future `question_hash` instead of raw question in metadata.
- **[Risk] Double configuration (app settings vs raw `LANGCHAIN_*`)** → Mitigation: startup helper overwrites env only when app tracing enabled; document precedence in README.
- **[Risk] Stream and non-stream traces hard to compare** → Mitigation: same metadata except `streaming` flag.
- **[Trade-off] No `/query` tracing in v1** → LangGraph users must wait or set env manually without unified metadata.

## Migration Plan

1. Add dependency and configuration defaults (tracing off).
2. Deploy tracing module and startup hook.
3. Instrument React then Plan-Execute services.
4. Operators enable tracing in `.env` with API key and project name.
5. Verify smoke runs appear in LangSmith UI.
6. Rollback: set `LANGSMITH_TRACING_ENABLED=false` and redeploy; no data migration.

## Open Questions

- Whether to generate and propagate `request_id` from FastAPI middleware in this change or a follow-up.
- Whether to include question text in metadata or only a hash/redacted preview for privacy.
