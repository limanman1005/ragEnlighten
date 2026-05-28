## ADDED Requirements

### Requirement: Evaluation outputs include trace correlation metadata
When LangSmith tracing is enabled, the system SHALL expose enough trace correlation metadata in evaluation outputs to support failure triage without changing agent API response schemas.

#### Scenario: Correlation fields are captured for traced runs
- **WHEN** an evaluation case executes while LangSmith tracing is effectively enabled
- **THEN** evaluation output includes route, endpoint, and phase metadata for the run
- **AND** includes available trace or run identifiers needed to locate the request in LangSmith

#### Scenario: Evaluation remains functional without LangSmith export
- **WHEN** LangSmith tracing is disabled or not exportable
- **THEN** evaluation outputs still include local case, mode, and response-derived analysis fields
- **AND** missing LangSmith identifiers do not fail the evaluation run
