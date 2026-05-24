import unittest

from app.services.web_search import (
    MockWebSearchProvider,
    WebSearchResult,
    format_web_search_results,
    run_web_search,
    search_web,
    web_results_to_documents,
)


class WebSearchProviderTests(unittest.TestCase):
    def test_mock_provider_returns_deterministic_limited_results(self):
        provider = MockWebSearchProvider()

        results = provider.search("current rag news", top_k=2)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].title, "Mock web result 1 for current rag news")
        self.assertEqual(results[0].url, "https://example.com/search/current-rag-news/1")
        self.assertIn("current rag news", results[0].snippet)

    def test_search_web_uses_mock_provider_without_credentials(self):
        results = search_web("agent tools", top_k=1, provider_name="mock")

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].title, "Mock web result 1 for agent tools")

    def test_formatting_and_documents_preserve_web_evidence(self):
        results = [
            WebSearchResult(
                title="Agent search result",
                url="https://example.com/agent-search",
                snippet="A concise result snippet.",
            )
        ]

        output = format_web_search_results(results)
        documents = web_results_to_documents(results, query="agent search")

        self.assertIn("title=Agent search result", output)
        self.assertIn("url=https://example.com/agent-search", output)
        self.assertEqual(documents[0].metadata["source"], "https://example.com/agent-search")
        self.assertEqual(documents[0].metadata["title"], "Agent search result")
        self.assertEqual(documents[0].metadata["source_type"], "web")
        self.assertEqual(documents[0].metadata["retrieval_query"], "agent search")
        self.assertEqual(documents[0].page_content, "A concise result snippet.")

    def test_run_web_search_contains_unsupported_provider_errors(self):
        output, documents = run_web_search("agent tools", provider_name="unsupported")

        self.assertIn("Web search failed", output)
        self.assertEqual(documents, [])


if __name__ == "__main__":
    unittest.main()
