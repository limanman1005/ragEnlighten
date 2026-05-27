## 1. Tracing helpers and configuration

- [x] 1.1 Add `AgentTraceContext` and helpers to propagate parent run + unified metadata in `app/core/tracing.py`
- [x] 1.2 Update `configure_langsmith_tracing` and `is_tracing_enabled` to require API key for effective tracing; warn when key missing
- [x] 1.3 Remove `LANGCHAIN_TRACING_V2` from read aliases of `langsmith_tracing_enabled` in `app/core/config.py`
- [x] 1.4 Rename shadowed `config` variable to `run_config` in `build_run_config`
- [x] 1.5 Extend `tests/test_tracing.py` for effective tracing guard and metadata on child configs

## 2. Plan-Execute trace hierarchy

- [x] 2.1 Pass `AgentTraceContext` through planning, execution, and synthesis; nest runs under root `trace()`
- [x] 2.2 Apply full unified metadata to `plan-execute-execution` and `plan-execute-step-*` traces
- [x] 2.3 Fix unknown/skipped plan tools to avoid false-success step traces
- [x] 2.4 Add or update unit tests for step metadata and skipped-tool tracing behavior

## 3. React Agent trace hierarchy

- [x] 3.1 Wrap non-stream and stream entry points with root `trace()` and child `build_run_config(trace_context=...)`
- [x] 3.2 Verify agent runs nest under root in LangSmith (smoke note in README)

## 4. Documentation

- [x] 4.1 Update `.env.example` and README: `LANGSMITH_*` controls enablement; API key required; hierarchical trace expectation
