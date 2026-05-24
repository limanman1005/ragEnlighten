## Why

The current React Agent endpoint follows a ReAct loop where the model decides, calls tools, and revises step by step during execution. Some callers need a more auditable question-answering mode where the agent first produces an explicit plan, then executes that fixed plan so the workflow is easier to inspect, debug, and present in the UI.

## What Changes

- Add a new question-answering Agent API that separates planning from execution.
- The new Agent will first generate a structured plan for the user question, including intended steps and tool usage.
- The Agent will then execute the generated plan and return the answer, plan, tool calls, retrieved sources, trace, debug events, confidence, and validation details.
- Add a streaming variant using SSE so clients can observe planning, execution progress, tokens, and final response.
- Keep existing `/query`, `/chat/react-agent`, and `/chat/react-agent/stream` behavior intact.

## Capabilities

### New Capabilities

- `plan-execute-agent-api`: Provides a plan-first question-answering Agent API with non-streaming and SSE streaming responses.

### Modified Capabilities

None.

## Impact

- Adds new FastAPI chat endpoints under `app/api/routes.py`.
- Adds a plan-execute Agent service alongside the existing ReAct Agent service.
- Reuses existing LangChain LLM, Chroma vector store, document source formatting, validation response models, and SSE formatting helper.
- Updates Streamlit only if the UI needs a selectable mode for the new Agent endpoint.
- Adds tests for planning-before-execution behavior and SSE response formatting.
