"""Shared web search tool support for Agent workflows."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Protocol

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


Transport = Callable[[str, dict[str, object], dict[str, str], float], dict[str, Any]]


class TavilyWebSearchProvider:
    """Tavily Search API provider for live web evidence."""

    endpoint = "https://api.tavily.com/search"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        timeout_seconds: float | None = None,
        search_depth: str | None = None,
        include_raw_content: bool | None = None,
        max_raw_content_chars: int | None = None,
        transport: Transport | None = None,
    ) -> None:
        self.api_key = api_key if api_key is not None else settings.web_search_api_key
        self.timeout_seconds = (
            timeout_seconds if timeout_seconds is not None else settings.web_search_timeout_seconds
        )
        self.search_depth = search_depth or settings.web_search_tavily_search_depth
        self.include_raw_content = (
            include_raw_content
            if include_raw_content is not None
            else settings.web_search_tavily_include_raw_content
        )
        self.max_raw_content_chars = (
            max_raw_content_chars
            if max_raw_content_chars is not None
            else settings.web_search_tavily_max_raw_content_chars
        )
        self.transport = transport or _post_json

    def search(self, query: str, top_k: int) -> list[WebSearchResult]:
        if not self.api_key.strip():
            raise ValueError("WEB_SEARCH_API_KEY is required when WEB_SEARCH_PROVIDER=tavily")

        normalized_query = query.strip()
        if not normalized_query:
            return []

        payload: dict[str, object] = {
            "query": normalized_query,
            "max_results": max(0, top_k),
            "search_depth": self.search_depth,
            "include_raw_content": self.include_raw_content,
        }
        response = self.transport(
            self.endpoint,
            payload,
            {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            float(self.timeout_seconds),
        )
        raw_results = response.get("results")
        if not isinstance(raw_results, list):
            raise ValueError("Tavily response missing results list")

        results: list[WebSearchResult] = []
        for item in raw_results:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or "").strip()
            url = str(item.get("url") or "").strip()
            snippet = str(item.get("content") or "").strip()
            if not snippet:
                snippet = str(item.get("raw_content") or "").strip()[: max(0, self.max_raw_content_chars)]
            if title and url and snippet:
                results.append(WebSearchResult(title=title, url=url, snippet=snippet))
        return results


def _post_json(
    url: str,
    payload: dict[str, object],
    headers: dict[str, str],
    timeout: float,
) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Tavily HTTP error {exc.code}: {exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Tavily request failed: {exc.reason}") from exc

    parsed = json.loads(body)
    if not isinstance(parsed, dict):
        raise ValueError("Tavily response must be a JSON object")
    return parsed


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "query"


def get_web_search_provider(provider_name: str | None = None) -> WebSearchProvider:
    provider = (provider_name or settings.web_search_provider).strip().lower()
    if provider == "mock":
        return MockWebSearchProvider()
    if provider == "tavily":
        return TavilyWebSearchProvider()
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
