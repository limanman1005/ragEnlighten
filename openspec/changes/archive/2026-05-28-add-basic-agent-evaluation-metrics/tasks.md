## 1. Evaluation baseline scaffolding

- [x] 1.1 Create evaluation module layout (dataset, runner, scoring, exporters) under a dedicated directory.
- [x] 1.2 Add a baseline dataset file format and seed representative cases for KB-grounded, web-required, and no-evidence scenarios.
- [x] 1.3 Document dataset schema fields (`case_id`, inputs, expected behavior labels) and validation rules.

## 2. Batch execution pipeline

- [x] 2.1 Implement a batch runner that executes each dataset case against both ReAct and Plan-Execute endpoints.
- [x] 2.2 Capture per-case/per-mode execution outputs, including response payloads and request failure states.
- [x] 2.3 Add configurable endpoint base URL, timeout, and mode selection flags for local and CI usage.

## 3. Baseline metric scoring

- [x] 3.1 Implement tool expectation scoring from expected tool labels versus observed `tool_calls`.
- [x] 3.2 Implement grounding pass scoring using `validation` and source/citation-related response fields.
- [x] 3.3 Implement no-evidence answer leakage scoring for cases that require guarded refusal.
- [x] 3.4 Implement request success rate scoring that accounts for transport/runtime failures.

## 4. Analysis artifacts and LangSmith correlation

- [x] 4.1 Export detailed per-case results (`results.json`) with expected vs observed score flags.
- [x] 4.2 Export aggregate per-mode summaries (`summary.json`) with baseline metric rates for ReAct and Plan-Execute.
- [x] 4.3 Add optional CSV export for manual review workflows.
- [x] 4.4 Include LangSmith trace correlation metadata when tracing is enabled, while keeping evaluation functional without LangSmith.

## 5. Quality checks and developer workflow

- [x] 5.1 Add automated tests for dataset parsing, scoring rules, and summary aggregation.
- [x] 5.2 Add an end-to-end smoke test that runs a tiny fixture dataset through both modes using mocked responses.
- [x] 5.3 Update README with commands to run baseline evaluation locally and interpret output metrics.
- [x] 5.4 Verify OpenSpec artifact completeness and ensure the change is apply-ready.
