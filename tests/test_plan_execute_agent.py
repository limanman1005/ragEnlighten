import asyncio
import unittest

from app.services.plan_execute_agent import (
    PlanStep,
    PlanExecutionResult,
    _apply_grounding_guard,
    _build_fallback_plan,
    _build_validation,
    _execute_plan_steps,
    _parse_plan_steps,
)


class PlanParsingTests(unittest.TestCase):
    def test_parses_json_plan_steps(self):
        raw = """
        {
          "steps": [
            {
              "id": "1",
              "purpose": "Find relevant resume evidence",
              "tool": "knowledge_base_search",
              "query": "candidate project experience"
            },
            {
              "id": "2",
              "purpose": "Summarize evidence",
              "tool": "answer_synthesis",
              "instruction": "Answer from retrieved evidence"
            }
          ]
        }
        """

        steps = _parse_plan_steps(raw, "What projects did this candidate do?")

        self.assertEqual(
            steps,
            [
                PlanStep(
                    step_id="1",
                    purpose="Find relevant resume evidence",
                    tool="knowledge_base_search",
                    query="candidate project experience",
                ),
                PlanStep(
                    step_id="2",
                    purpose="Summarize evidence",
                    tool="answer_synthesis",
                    query="Answer from retrieved evidence",
                ),
            ],
        )

    def test_falls_back_to_retrieval_plan_for_malformed_model_output(self):
        steps = _parse_plan_steps("not json", "Summarize the indexed profile")

        self.assertEqual(steps, _build_fallback_plan("Summarize the indexed profile"))
        self.assertEqual(steps[0].tool, "knowledge_base_search")
        self.assertEqual(steps[-1].tool, "answer_synthesis")


class PlanExecutionOrderingTests(unittest.TestCase):
    def test_execution_starts_after_plan_steps_are_available(self):
        calls: list[str] = []
        steps = [
            PlanStep("1", "Search first", "knowledge_base_search", "alpha"),
            PlanStep("2", "Read metadata second", "collection_overview", ""),
        ]

        async def run():
            calls.append("plan_ready")

            def search(query: str):
                calls.append(f"search:{query}")
                return "search output", []

            def overview():
                calls.append("overview")
                return "overview output"

            await _execute_plan_steps(steps, search=search, overview=overview)

        asyncio.run(run())

        self.assertEqual(calls, ["plan_ready", "search:alpha", "overview"])


class PlanGroundingTests(unittest.TestCase):
    def test_metadata_tool_output_counts_as_supporting_context(self):
        execution = PlanExecutionResult(
            documents=[],
            tool_calls=[
                {
                    "name": "collection_overview",
                    "status": "success",
                    "input_summary": "",
                    "output_summary": "Collections: docs(3)",
                }
            ],
            tool_context=["Collections: docs(3)"],
            trace=[],
            debug_events=[],
        )

        validation = _build_validation(execution)
        answer, needs_review, reason = _apply_grounding_guard("Collections: docs(3)", execution)

        self.assertTrue(validation["passed"])
        self.assertTrue(validation["citations_verified"])
        self.assertEqual(answer, "Collections: docs(3)")
        self.assertFalse(needs_review)
        self.assertIsNone(reason)


if __name__ == "__main__":
    unittest.main()
