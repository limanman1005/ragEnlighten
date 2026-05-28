## Why

Current LangSmith integration in this project focuses on observability (trace export, hierarchy, and metadata), but the team still lacks a repeatable baseline evaluation workflow to compare ReAct and Plan-Execute behavior over time. A minimal, deterministic evaluation baseline is needed now to prevent regressions and make prompt/tool strategy changes measurable.

## What Changes

- Add a baseline agent evaluation workflow that runs the same test dataset against both ReAct and Plan-Execute endpoints.
- Define a compact evaluation dataset schema and seed examples for core scenarios: KB-grounded, web-required, and no-evidence guard behavior.
- Define basic rule-based metrics and summary outputs (JSON/CSV) for batch comparison:
  - tool expectation hit rate
  - grounding pass rate
  - unsupported factual answer rate (no-evidence answer leakage)
- Provide an executable evaluation script and documentation for local runs and CI-friendly usage.
- Persist per-run details needed for failure triage (question id, endpoint mode, expected vs actual tool usage, validation flags, and trace references when available).

## Capabilities

### New Capabilities
- `agent-evaluation-baseline`: Baseline offline evaluation dataset, metrics, and batch comparison workflow for ReAct and Plan-Execute agents.

### Modified Capabilities
- `agent-langsmith-observability`: Clarify how LangSmith trace references are attached to evaluation outputs for failure analysis without requiring schema changes to agent APIs.

## Impact

- Affected code:
  - new evaluation scripts and dataset files (for example under `evals/` and `scripts/`)
  - optional small wiring in agent response handling to expose trace identifiers when available
  - README updates for evaluation usage
- APIs:
  - no external API contract changes for existing agent endpoints
- Dependencies:
  - can reuse existing stack; optional LangSmith client usage remains behind configuration
- Systems:
  - local developer workflow and CI can run baseline regression checks before prompt/tool updates are merged
