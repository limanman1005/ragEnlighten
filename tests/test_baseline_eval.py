import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evals.baseline import (
    PLAN_EXECUTE_MODE,
    REACT_MODE,
    EvalCase,
    EvalOutcome,
    build_summary,
    load_cases,
    run_cases_http,
    score_outcomes,
)


class DatasetParsingTests(unittest.TestCase):
    def test_load_cases_parses_required_and_optional_fields(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_path = Path(tmp_dir) / "cases.jsonl"
            dataset_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "case_id": "c1",
                                "question": "q1",
                                "expected_tools": ["knowledge_base_search"],
                                "should_be_grounded": True,
                            },
                            ensure_ascii=False,
                        ),
                        json.dumps(
                            {
                                "case_id": "c2",
                                "question": "q2",
                                "history": [{"role": "user", "content": "old"}],
                                "should_refuse_when_no_evidence": True,
                            },
                            ensure_ascii=False,
                        ),
                    ]
                ),
                encoding="utf-8",
            )
            cases = load_cases(dataset_path)

        self.assertEqual(len(cases), 2)
        self.assertEqual(cases[0].expected_tools, ["knowledge_base_search"])
        self.assertTrue(cases[1].should_refuse_when_no_evidence)
        self.assertEqual(cases[1].history[0]["role"], "user")


class ScoringAndSummaryTests(unittest.TestCase):
    def test_score_outcomes_and_summary(self):
        case = EvalCase(
            case_id="case-1",
            question="hello",
            expected_tools=["knowledge_base_search"],
            should_be_grounded=True,
            should_refuse_when_no_evidence=True,
        )
        success_response = {
            "route": "react_agent",
            "tool_calls": [{"name": "knowledge_base_search"}],
            "validation": {"passed": True, "citations_verified": True, "issues": []},
            "sources": [{"source": "doc"}],
            "answer": "grounded answer",
        }
        failed_response = {
            "route": "plan_execute_agent",
            "tool_calls": [],
            "validation": {"passed": False, "citations_verified": False, "issues": ["missing evidence"]},
            "sources": [],
            "answer": "This is a made-up factual answer",
        }

        scored = score_outcomes(
            [
                EvalOutcome(
                    case=case,
                    mode=REACT_MODE,
                    endpoint="/api/v1/chat/react-agent",
                    success=True,
                    status_code=200,
                    duration_ms=40,
                    response=success_response,
                ),
                EvalOutcome(
                    case=case,
                    mode=PLAN_EXECUTE_MODE,
                    endpoint="/api/v1/chat/plan-execute-agent",
                    success=True,
                    status_code=200,
                    duration_ms=70,
                    response=failed_response,
                ),
            ]
        )
        summary = build_summary(scored)

        react_item = next(item for item in scored if item.mode == REACT_MODE)
        plan_item = next(item for item in scored if item.mode == PLAN_EXECUTE_MODE)

        self.assertTrue(react_item.tool_expectation_hit)
        self.assertFalse(plan_item.tool_expectation_hit)
        self.assertFalse(plan_item.grounding_pass)
        self.assertTrue(plan_item.no_evidence_answer_leakage)
        self.assertEqual(len(summary["modes"]), 2)


class BatchRunnerSmokeTests(unittest.TestCase):
    def test_run_cases_http_executes_both_modes_with_mocked_endpoint(self):
        cases = [EvalCase(case_id="smoke-1", question="q")]

        def fake_call(case, *, mode, endpoint, url, timeout_seconds):
            return EvalOutcome(
                case=case,
                mode=mode,
                endpoint=endpoint,
                success=True,
                status_code=200,
                duration_ms=12,
                response={
                    "route": "react_agent" if mode == REACT_MODE else "plan_execute_agent",
                    "tool_calls": [{"name": "knowledge_base_search"}],
                    "validation": {"passed": True, "citations_verified": True, "issues": []},
                    "sources": [{"source": "doc"}],
                    "answer": "ok",
                },
            )

        with patch("evals.baseline._call_endpoint", side_effect=fake_call) as mocked_call:
            outcomes = run_cases_http(
                cases,
                base_url="http://127.0.0.1:8000",
                timeout_seconds=5.0,
                modes=[REACT_MODE, PLAN_EXECUTE_MODE],
            )

        self.assertEqual(len(outcomes), 2)
        self.assertEqual(mocked_call.call_count, 2)
        self.assertEqual({item.mode for item in outcomes}, {REACT_MODE, PLAN_EXECUTE_MODE})


if __name__ == "__main__":
    unittest.main()

