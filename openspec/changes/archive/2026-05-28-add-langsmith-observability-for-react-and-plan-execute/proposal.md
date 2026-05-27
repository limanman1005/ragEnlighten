## Why

The project exposes two distinct agent question-answering flows—ReAct (`/chat/react-agent`) and plan-first (`/chat/plan-execute-agent`)—but troubleshooting and quality review still rely primarily on application logs and `debug_events`. Operators cannot easily compare runs, inspect LLM/tool chains, or filter failures by route or collection in a unified observability platform. LangSmith tracing should be added now so both agent styles share a consistent trace contract before building offline evaluation datasets.

## What Changes

- Add LangSmith configuration (enable flag, API key, project, optional endpoint) and startup initialization for LangChain-compatible tracing environment variables.
- Introduce a shared tracing helper module for building `RunnableConfig` with unified `run_name`, `tags`, and `metadata`.
- Instrument React Agent non-streaming and streaming paths (`ainvoke`, `astream_events`) with LangSmith run metadata.
- Instrument Plan-Execute Agent planning and synthesis LangChain chains with phase-level run metadata.
- Add recommended Plan-Execute execution-step observability so plan steps are visible and mappable to `tool_calls` in traces.
- Document configuration in `.env.example` and README; add `langsmith` dependency.
- Keep all existing API request/response contracts unchanged (`QueryResponse`, SSE event shapes, business semantics).

## Capabilities

### New Capabilities

- `agent-langsmith-observability`: Unified LangSmith tracing for React and Plan-Execute agent APIs (non-stream and stream), including metadata schema, run naming, and phase/step visibility requirements.

### Modified Capabilities

None.

## Impact

- `app/core/config.py`: LangSmith settings.
- `app/core/tracing.py` (new): tracing bootstrap and run config helpers.
- `app/main.py`: call tracing configuration at application startup.
- `app/services/react_agent.py`: pass run config to agent invoke/stream calls.
- `app/services/plan_execute_agent.py`: pass run config to planning/synthesis chains; optional step-level instrumentation for execution loop.
- `requirements.txt`: add `langsmith`.
- `.env.example` and README: configuration and troubleshooting notes.
- Tests: tracing helper unit tests and smoke verification guidance (no change to agent answer logic).
