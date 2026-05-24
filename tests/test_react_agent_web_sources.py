import unittest

from app.services.react_agent import _build_react_agent_result
from app.services.web_search import WebSearchResult, web_results_to_documents


class ReactAgentWebSourceTests(unittest.TestCase):
    def test_web_search_documents_count_as_grounded_sources(self):
        documents = web_results_to_documents(
            [
                WebSearchResult(
                    title="Agent web search",
                    url="https://example.com/agent-web-search",
                    snippet="External evidence for the answer.",
                )
            ],
            query="agent web search",
        )

        result = _build_react_agent_result(
            question="What changed recently?",
            documents=documents,
            tool_calls=[
                {
                    "name": "web_search",
                    "status": "success",
                    "input_summary": "agent web search",
                    "output_summary": "External evidence for the answer.",
                }
            ],
            trace=["1. Tool web_search returned a result"],
            answer="External evidence for the answer.",
        )

        self.assertTrue(result["validation"]["passed"])
        self.assertTrue(result["validation"]["citations_verified"])
        self.assertEqual(result["tool_calls"][0]["name"], "web_search")
        self.assertEqual(result["documents"][0].metadata["source"], "https://example.com/agent-web-search")


if __name__ == "__main__":
    unittest.main()
