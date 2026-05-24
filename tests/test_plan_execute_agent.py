import asyncio
import unittest

from app.services.plan_execute_agent import (
    Document,
    PlanStep,
    PlanExecutionResult,
    _apply_grounding_guard,
    _build_fallback_plan,
    _build_validation,
    _execute_plan_steps,
    _normalize_plan_tool,
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

    def test_parses_web_search_plan_step(self):
        raw = """
        {
          "steps": [
            {
              "id": "1",
              "purpose": "Find current external evidence",
              "tool": "internet_search",
              "query": "latest RAG evaluation methods"
            }
          ]
        }
        """

        steps = _parse_plan_steps(raw, "What changed recently?")

        self.assertEqual(steps[0].tool, "web_search")
        self.assertEqual(steps[0].query, "latest RAG evaluation methods")
        self.assertEqual(steps[-1].tool, "answer_synthesis")

    def test_normalizes_web_search_aliases(self):
        self.assertEqual(_normalize_plan_tool("web"), "web_search")
        self.assertEqual(_normalize_plan_tool("internet_search"), "web_search")
        self.assertEqual(_normalize_plan_tool("web_search"), "web_search")


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

    def test_executes_web_search_steps_in_plan_order(self):
        calls: list[str] = []
        steps = [
            PlanStep("1", "Search web first", "web_search", "external topic"),
            PlanStep("2", "Synthesize answer", "answer_synthesis", ""),
        ]

        async def run():
            def search(query: str):
                calls.append(f"kb:{query}")
                return "knowledge output", []

            def overview():
                calls.append("overview")
                return "overview output"

            def web_search(query: str):
                calls.append(f"web:{query}")
                return "web output", [
                    Document(
                        page_content="web evidence",
                        metadata={"source": "https://example.com/web", "source_type": "web"},
                    )
                ]

            return await _execute_plan_steps(
                steps,
                search=search,
                overview=overview,
                web_search=web_search,
            )

        execution = asyncio.run(run())

        self.assertEqual(calls, ["web:external topic"])
        self.assertEqual(execution.tool_context, ["web output"])
        self.assertEqual(len(execution.documents), 1)
        self.assertEqual(execution.tool_calls[0]["name"], "web_search")
        self.assertTrue(
            any("Tool web_search returned a result" in item for item in execution.trace)
        )

    def test_web_search_failure_is_recorded_without_supporting_context(self):
        steps = [PlanStep("1", "Search web", "web_search", "external topic")]

        async def run():
            def search(query: str):
                return "knowledge output", []

            def overview():
                return "overview output"

            def web_search(query: str):
                raise ValueError("provider unavailable")

            return await _execute_plan_steps(
                steps,
                search=search,
                overview=overview,
                web_search=web_search,
            )

        execution = asyncio.run(run())

        self.assertEqual(execution.tool_context, [])
        self.assertEqual(execution.tool_calls[0]["name"], "web_search")
        self.assertEqual(execution.tool_calls[0]["status"], "failed")
        self.assertIn("provider unavailable", execution.tool_calls[0]["output_summary"])


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

    def test_empty_web_search_output_does_not_count_as_supporting_context(self):
        execution = PlanExecutionResult(
            documents=[],
            tool_calls=[
                {
                    "name": "web_search",
                    "status": "success",
                    "input_summary": "current topic",
                    "output_summary": "No web search results were found.",
                }
            ],
            tool_context=[],
            trace=[],
            debug_events=[],
        )

        validation = _build_validation(execution)
        answer, needs_review, reason = _apply_grounding_guard("Unsupported answer", execution)

        self.assertFalse(validation["passed"])
        self.assertFalse(validation["citations_verified"])
        self.assertNotEqual(answer, "Unsupported answer")
        self.assertTrue(needs_review)
        self.assertIsNotNone(reason)


if __name__ == "__main__":
    unittest.main()
