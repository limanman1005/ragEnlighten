## MODIFIED Requirements

### Requirement: Explicit planning before execution

The system SHALL complete an explicit planning phase before performing retrieval, metadata lookup, web search, or final answer synthesis for the plan-execute Agent.

#### Scenario: Plan is returned before execution details

- **WHEN** a client calls the plan-execute Agent
- **THEN** the response `plan` contains ordered human-readable steps created before the recorded execution `tool_calls`

#### Scenario: Execution follows the generated plan

- **WHEN** the planning phase produces retrieval, metadata, or web search steps
- **THEN** the system executes those steps in plan order and records each executed tool call in `tool_calls`

#### Scenario: Plan can include web search

- **WHEN** the planning phase determines that the question needs external or current web information
- **THEN** the generated plan can include a `web_search` step before final answer synthesis
