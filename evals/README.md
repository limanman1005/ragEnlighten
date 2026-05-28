## Baseline Agent Evaluation Dataset

`baseline_cases.jsonl` uses one JSON object per line.

### Required fields

- `case_id`: Stable unique identifier for the case.
- `question`: User question sent to agent endpoints.

### Optional fields

- `collection_name`: Collection override passed to the endpoint.
- `history`: Prior messages list (same shape as chat history request).
- `expected_tools`: Expected tool names (`knowledge_base_search`, `web_search`, `collection_overview`).
- `should_be_grounded`: Expected grounding behavior (`true`/`false`).
- `should_refuse_when_no_evidence`: Whether an ungrounded factual answer should be treated as leakage.

### Validation rules

- `case_id` and `question` must be non-empty strings.
- `expected_tools` must be a list when provided.
- `history` must be a list when provided.
- Empty lines are ignored.

