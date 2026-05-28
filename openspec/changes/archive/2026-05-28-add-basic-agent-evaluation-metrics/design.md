## Context

The repository already supports two agent execution modes (`react_agent` and `plan_execute_agent`) and exports rich runtime signals such as `tool_calls`, `trace`, `validation`, and optional LangSmith traces. However, quality checks are still mostly endpoint-level tests and ad-hoc manual inspection. There is no baseline offline evaluation set that can be executed in batch to compare agent modes and detect regressions after prompt, tool, or routing changes.

The requested scope is a "basic class" evaluation baseline: small, deterministic, and easy to run locally, while still producing useful comparison metrics and failure triage artifacts.

## Goals / Non-Goals

**Goals:**
- Define a minimal evaluation dataset format that supports side-by-side ReAct vs Plan-Execute runs.
- Implement a batch evaluation runner that executes both modes on the same test cases.
- Compute baseline rule-based metrics from existing response fields (`tool_calls`, `validation`, `sources`, answer text).
- Output machine-readable run artifacts (per-case results + aggregated summary) suitable for CI and future dashboards.
- Link failures to observability context (trace metadata and optional LangSmith references).

**Non-Goals:**
- Building a full LLM-as-judge scoring framework in this change.
- Introducing new external API contracts for chat endpoints.
- Reworking agent orchestration logic or prompt strategies as part of this baseline.
- Enforcing CI blocking gates in the first iteration (reporting first, gating later).

## Decisions

### 1) Add a standalone offline evaluation package

Create a lightweight evaluation module (for example under `evals/`) with:
- dataset file(s) (`jsonl`)
- execution runner
- rule-based scorers
- result exporter (JSON + optional CSV)

Rationale: keeps evaluation logic isolated from request-serving code and minimizes risk to runtime paths.

Alternatives considered:
- Add evaluation logic inside existing API handlers: rejected due to coupling and operational risk.
- Use only LangSmith datasets immediately: rejected for baseline phase because local deterministic execution should work even when LangSmith is disabled.

### 2) Use rule-based metrics first, built from existing response contract

Baseline metrics:
- `tool_expectation_hit_rate`
- `grounding_pass_rate`
- `no_evidence_answer_rate`
- `request_success_rate`

Inputs use existing response fields:
- `tool_calls`
- `validation.passed`
- `validation.citations_verified`
- `sources`
- `answer`

Rationale: deterministic and low-cost scoring suitable for quick iteration.

Alternatives considered:
- LLM-as-judge only: deferred because it adds cost/variance and requires additional prompt tuning.

### 3) Evaluate both modes per case with a shared schema

Each dataset case includes:
- stable `case_id`
- `question`
- optional `collection_name` and `history`
- expected behavior labels (expected tools, should_be_grounded, should_refuse_when_no_evidence)

Runner executes:
- ReAct endpoint
- Plan-Execute endpoint

Rationale: enables direct comparability for each scenario and avoids mode-specific datasets too early.

Alternatives considered:
- Separate datasets by mode: rejected for baseline because cross-mode comparison is the primary goal.

### 4) Produce two-level output artifacts for triage

Per-run outputs:
- `results.json`: detailed per-case/per-mode outputs and metric flags
- `summary.json`: aggregated metrics and deltas between modes
- optional `results.csv` for quick spreadsheet review

Rationale: supports both automated processing and manual failure analysis.

Alternatives considered:
- Summary-only output: rejected because it hides actionable failure context.

### 5) Keep LangSmith integration optional but trace-aware

When LangSmith tracing is enabled, include trace identifiers/metadata references in evaluation outputs if available. When disabled, evaluation still works with local response data.

Rationale: maintains portability and avoids making evaluation dependent on external credentials.

Alternatives considered:
- Mandatory LangSmith dependency: rejected for local/offline workflow reliability.

## Risks / Trade-offs

- **[Risk] Dataset quality may be too narrow initially** → Mitigation: start with representative scenario buckets and add failing production-derived cases incrementally.
- **[Risk] Rule metrics may miss nuanced answer quality issues** → Mitigation: add optional LLM/human evaluators in a later phase.
- **[Risk] Endpoint/network instability can skew results** → Mitigation: capture transport errors separately and compute request success rate explicitly.
- **[Risk] Metric definitions may be interpreted inconsistently** → Mitigation: document metric formulas and expected-case labels in dataset schema docs.

## Migration Plan

1. Introduce dataset schema and seed baseline cases.
2. Implement evaluation runner and rule scorers.
3. Add result export and summary generation.
4. Document command usage in README.
5. (Optional follow-up) Add CI job that runs baseline evaluation and uploads artifacts.

Rollback:
- If issues occur, evaluation files can be removed without affecting existing APIs.
- Runtime behavior of chat endpoints remains unchanged because evaluation runs out-of-band.

## Open Questions

- Should baseline dataset include multilingual cases from day one, or start with Chinese-only scenarios and expand later?
- What threshold values should be recommended for future CI gating (informational in this change vs enforced later)?
- Should optional LangSmith trace IDs be extracted from response metadata or from SDK-level run callbacks in the evaluation runner?
