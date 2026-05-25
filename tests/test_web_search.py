import unittest
from unittest.mock import patch

from app.core.config import settings
from app.services.web_search import (
    MockWebSearchProvider,
    TavilyWebSearchProvider,
    WebSearchResult,
    format_web_search_results,
    get_web_search_provider,
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

    def test_provider_factory_selects_tavily(self):
        provider = get_web_search_provider("tavily")

        self.assertIsInstance(provider, TavilyWebSearchProvider)

    def test_tavily_provider_maps_results_and_honors_request_settings(self):
        captured: dict[str, object] = {}

        def transport(url: str, payload: dict[str, object], headers: dict[str, str], timeout: float):
            captured["url"] = url
            captured["payload"] = payload
            captured["headers"] = headers
            captured["timeout"] = timeout
            return {
                "results": [
                    {
                        "title": "Tavily result",
                        "url": "https://example.com/tavily",
                        "content": "Tavily result content.",
                    }
                ]
            }

        provider = TavilyWebSearchProvider(
            api_key="tvly-test",
            timeout_seconds=7.5,
            search_depth="advanced",
            include_raw_content=True,
            transport=transport,
        )

        results = provider.search("agent web search", top_k=2)

        self.assertEqual(results, [
            WebSearchResult(
                title="Tavily result",
                url="https://example.com/tavily",
                snippet="Tavily result content.",
            )
        ])
        self.assertEqual(captured["url"], "https://api.tavily.com/search")
        self.assertEqual(captured["timeout"], 7.5)
        self.assertEqual(captured["headers"]["Authorization"], "Bearer tvly-test")
        self.assertEqual(
            captured["payload"],
            {
                "query": "agent web search",
                "max_results": 2,
                "search_depth": "advanced",
                "include_raw_content": True,
            },
        )

    def test_run_web_search_contains_missing_tavily_key_errors(self):
        original_key = settings.web_search_api_key
        settings.web_search_api_key = ""
        try:
            output, documents = run_web_search("agent tools", provider_name="tavily")
        finally:
            settings.web_search_api_key = original_key

        self.assertIn("Web search failed", output)
        self.assertIn("WEB_SEARCH_API_KEY", output)
        self.assertEqual(documents, [])

    def test_tavily_provider_contains_http_errors(self):
        def transport(url: str, payload: dict[str, object], headers: dict[str, str], timeout: float):
            raise RuntimeError("429 Too Many Requests")

        provider = TavilyWebSearchProvider(api_key="tvly-test", transport=transport)

        with self.assertRaises(RuntimeError):
            provider.search("agent web search", top_k=1)

    def test_tavily_provider_invalid_response_returns_no_results(self):
        def transport(url: str, payload: dict[str, object], headers: dict[str, str], timeout: float):
            return {"unexpected": []}

        provider = TavilyWebSearchProvider(api_key="tvly-test", transport=transport)

        with self.assertRaises(ValueError):
            provider.search("agent web search", top_k=1)

    def test_tavily_provider_empty_results_returns_no_results(self):
        def transport(url: str, payload: dict[str, object], headers: dict[str, str], timeout: float):
            return {"results": []}

        provider = TavilyWebSearchProvider(api_key="tvly-test", transport=transport)

        self.assertEqual(provider.search("agent web search", top_k=1), [])

    def test_tavily_provider_bounds_raw_content_fallback(self):
        raw_content = "x" * 1200

        def transport(url: str, payload: dict[str, object], headers: dict[str, str], timeout: float):
            return {
                "results": [
                    {
                        "title": "Raw result",
                        "url": "https://example.com/raw",
                        "raw_content": raw_content,
                    }
                ]
            }

        provider = TavilyWebSearchProvider(
            api_key="tvly-test",
            include_raw_content=True,
            max_raw_content_chars=200,
            transport=transport,
        )

        results = provider.search("agent web search", top_k=1)

        self.assertEqual(len(results[0].snippet), 200)

    def test_run_web_search_contains_tavily_request_failures(self):
        original_key = settings.web_search_api_key
        settings.web_search_api_key = "tvly-test"
        try:
            with patch("app.services.web_search._post_json", side_effect=RuntimeError("timeout")):
                output, documents = run_web_search("agent tools", provider_name="tavily")
        finally:
            settings.web_search_api_key = original_key

        self.assertIn("Web search failed", output)
        self.assertIn("timeout", output)
        self.assertEqual(documents, [])

    def test_run_web_search_tavily_invalid_response_has_no_documents(self):
        original_key = settings.web_search_api_key
        settings.web_search_api_key = "tvly-test"
        try:
            with patch("app.services.web_search._post_json", return_value={"unexpected": []}):
                output, documents = run_web_search("agent tools", provider_name="tavily")
        finally:
            settings.web_search_api_key = original_key

        self.assertIn("Web search failed", output)
        self.assertEqual(documents, [])


if __name__ == "__main__":
    unittest.main()
