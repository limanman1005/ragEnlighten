## Requirements

### Requirement: Baseline evaluation dataset for dual agent modes
The system SHALL provide a baseline offline evaluation dataset format that can be executed against both ReAct and Plan-Execute agent endpoints.

#### Scenario: Dataset item defines comparable execution inputs
- **WHEN** an evaluation dataset item is loaded
- **THEN** it includes a stable case identifier and question input
- **AND** it supports optional collection name and history fields shared by both agent modes

#### Scenario: Dataset item includes expected behavior labels
- **WHEN** an evaluation dataset item is authored for baseline scoring
- **THEN** it can declare expected tool usage and grounding expectations
- **AND** it can declare whether no-evidence questions should trigger guarded refusal behavior

### Requirement: Batch regression execution across React and Plan-Execute
The system SHALL execute baseline evaluation cases in batch for both ReAct and Plan-Execute modes and collect per-case results.

#### Scenario: Runner executes both modes per case
- **WHEN** a baseline evaluation run starts
- **THEN** each dataset case is executed against ReAct and Plan-Execute endpoints
- **AND** per-mode request and response outputs are persisted for comparison

#### Scenario: Runner records transport and runtime failures
- **WHEN** an endpoint call fails due to timeout, transport error, or runtime exception
- **THEN** the run records the failure status for that case and mode
- **AND** aggregate metrics include the failure in request success calculations

### Requirement: Baseline rule-based metrics
The system SHALL compute deterministic baseline metrics from existing agent response fields without requiring LLM-as-judge scoring.

#### Scenario: Tool expectation hit rate is computed
- **WHEN** a case declares expected tool names
- **THEN** scoring compares expected tools with observed `tool_calls`
- **AND** aggregate output reports tool expectation hit rate per mode

#### Scenario: Grounding pass rate is computed
- **WHEN** evaluation responses include validation and source fields
- **THEN** scoring computes grounding pass status from response validation and citation evidence fields
- **AND** aggregate output reports grounding pass rate per mode

#### Scenario: No-evidence answer leakage is computed
- **WHEN** a case is labeled as requiring refusal when evidence is absent
- **THEN** scoring marks leakage if the response returns an unsupported factual answer without grounding evidence
- **AND** aggregate output reports no-evidence answer rate per mode

### Requirement: Evaluation run artifacts for analysis
The system SHALL emit machine-readable per-case and aggregate evaluation artifacts suitable for regression tracking.

#### Scenario: Per-case results are exported
- **WHEN** a batch evaluation run completes
- **THEN** the system writes per-case outputs containing case id, mode, expected labels, observed tool usage, validation flags, and score flags

#### Scenario: Aggregate summary is exported
- **WHEN** a batch evaluation run completes
- **THEN** the system writes aggregate metrics including request success rate, tool expectation hit rate, grounding pass rate, and no-evidence answer rate
- **AND** the summary distinguishes ReAct and Plan-Execute mode results
