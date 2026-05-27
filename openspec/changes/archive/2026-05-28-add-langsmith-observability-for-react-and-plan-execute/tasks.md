## 1. Configuration and dependencies

- [x] 1.1 Add `langsmith` to `requirements.txt`
- [x] 1.2 Add LangSmith settings to `app/core/config.py` with `LANGSMITH_*` and `LANGCHAIN_*` aliases
- [x] 1.3 Document LangSmith variables in `.env.example` (disabled by default)

## 2. Tracing module and startup

- [x] 2.1 Create `app/core/tracing.py` with `configure_langsmith_tracing`, `build_run_config`, and `merge_run_configs`
- [x] 2.2 Call `configure_langsmith_tracing()` from `app/main.py` at application startup
- [x] 2.3 Add unit tests for tracing helpers (enabled/disabled config, metadata merge)

## 3. React Agent instrumentation

- [x] 3.1 Pass tracing config to `agent.ainvoke` in `run_react_agent_query` with `react-agent-query` run name and unified metadata
- [x] 3.2 Pass tracing config to `agent.astream_events` in `stream_react_agent_query` with `react-agent-stream` run name and `streaming=true`

## 4. Plan-Execute Agent instrumentation

- [x] 4.1 Pass tracing config to planning chain invoke with `plan-execute-planning` and `phase=planning`
- [x] 4.2 Pass tracing config to synthesis chain invoke with `plan-execute-synthesis` and `phase=synthesis`
- [x] 4.3 Add root run metadata for `run_plan_execute_agent_query` and stream entry (`plan-execute-query` / `plan-execute-stream`)
- [x] 4.4 Instrument `_execute_plan_steps` with step-level trace metadata (`step_id`, `tool_name`, `step_status`, latency) aligned with `tool_calls`

## 5. Documentation and verification

- [x] 5.1 Add README section for enabling LangSmith, required env vars, and how to filter runs by `route` and `streaming`
- [x] 5.2 Run smoke requests against React and Plan-Execute endpoints with tracing enabled and confirm runs appear in LangSmith UI
- [x] 5.3 Confirm tracing disabled leaves agent responses and SSE behavior unchanged
