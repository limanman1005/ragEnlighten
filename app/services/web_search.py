"""Shared web search tool support for Agent workflows."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Protocol

from app.core.config import settings

try:
    from langchain_core.documents import Document
except ModuleNotFoundError:
    @dataclass
    class Document:  # type: ignore[no-redef]
        page_content: str
        metadata: dict[str, Any]


@dataclass(frozen=True)
class WebSearchResult:
    title: str
    url: str
    snippet: str


class WebSearchProvider(Protocol):
    def search(self, query: str, top_k: int) -> list[WebSearchResult]:
        """Return web search results for a query."""


class MockWebSearchProvider:
    """Deterministic provider used until a live search provider is configured."""

    def search(self, query: str, top_k: int) -> list[WebSearchResult]:
        normalized_query = query.strip()
        if not normalized_query:
            return []

        limit = max(0, top_k)
        slug = _slugify(normalized_query)
        return [
            WebSearchResult(
                title=f"Mock web result {index} for {normalized_query}",
                url=f"https://example.com/search/{slug}/{index}",
                snippet=(
                    f"Mock web search snippet {index} for '{normalized_query}'. "
                    "Replace the mock provider with a live provider for real web evidence."
                ),
            )
            for index in range(1, limit + 1)
        ]


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "query"


def get_web_search_provider(provider_name: str | None = None) -> WebSearchProvider:
    provider = (provider_name or settings.web_search_provider).strip().lower()
    if provider == "mock":
        return MockWebSearchProvider()
    raise ValueError(f"Unsupported web search provider: {provider}")


def search_web(
    query: str,
    top_k: int | None = None,
    provider_name: str | None = None,
) -> list[WebSearchResult]:
    if not settings.web_search_enabled:
        return []

    provider = get_web_search_provider(provider_name)
    result_limit = settings.web_search_top_k if top_k is None else top_k
    return provider.search(query, result_limit)


def format_web_search_results(results: list[WebSearchResult]) -> str:
    if not results:
        return "No web search results were found."

    return "\n\n".join(
        f"[{index}] title={result.title}; url={result.url}\n{result.snippet}"
        for index, result in enumerate(results, start=1)
    )


def web_results_to_documents(
    results: list[WebSearchResult],
    *,
    query: str,
) -> list[Document]:
    return [
        Document(
            page_content=result.snippet,
            metadata={
                "source": result.url,
                "source_type": "web",
                "title": result.title,
                "content_preview": result.snippet,
                "retrieval_hop": 1,
                "retrieval_query": query,
            },
        )
        for result in results
    ]


def run_web_search(
    query: str,
    provider_name: str | None = None,
) -> tuple[str, list[Document]]:
    try:
        results = search_web(query, provider_name=provider_name)
    except Exception as exc:
        return f"Web search failed: {exc}", []
    return format_web_search_results(results), web_results_to_documents(results, query=query)
