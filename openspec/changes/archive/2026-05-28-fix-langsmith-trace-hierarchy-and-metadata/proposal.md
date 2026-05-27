## Why

The initial LangSmith integration for React and Plan-Execute agents is functional but produces fragmented traces in LangSmith: planning, synthesis, and execution steps often appear as sibling runs instead of a single request hierarchy, execution-step metadata omits fields needed for filtering (`route`, `endpoint`, `collection_name`), and enabling tracing without an API key can silently fail. These gaps were identified in post-implementation code review and block reliable per-request debugging in LangSmith.

## What Changes

- Nest Plan-Execute planning, execution, and synthesis runs under a single root trace per API request (non-stream and stream).
- Propagate unified agent metadata to execution wrapper and per-step traces, not only to LLM chains.
- Validate tracing configuration at startup: warn and skip forcing `LANGCHAIN_TRACING_V2` when tracing is enabled but API key is missing.
- Clarify configuration precedence in docs: application `LANGSMITH_*` flags control tracing; `LANGCHAIN_*` env vars are set by the app when enabled (read alias for enable flag no longer includes `LANGCHAIN_TRACING_V2`).
- Fix Plan-Execute unknown or unsupported plan tools so step traces are not recorded as false successes.
- Add tests for metadata propagation, startup guard behavior, and parent-child trace wiring (mocked where needed).

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `agent-langsmith-observability`: Strengthen requirements for hierarchical traces, full metadata on execution steps, startup credential guard, and accurate step status for skipped tools.

## Impact

- `app/core/tracing.py`: request trace context helper, parent run propagation, startup guard, config alias adjustment.
- `app/core/config.py`: narrow `langsmith_tracing_enabled` env aliases.
- `app/services/plan_execute_agent.py`: pass trace context through planning, execution, synthesis; fix unknown-tool step handling.
- `app/services/react_agent.py`: optional root trace wrapper with child LangChain config (if needed for hierarchy).
- `tests/test_tracing.py` and Plan-Execute tracing tests: new cases.
- `.env.example` and README: configuration precedence and troubleshooting notes.
