import json
import unittest
from unittest.mock import MagicMock, patch

from pydantic import ValidationError

from app.services.agent_tools import (
    AgentToolResult,
    KnowledgeBaseSearchRequest,
    ToolSourceMetadata,
    apply_tool_output_to_documents,
    build_retrieval_filter,
    document_to_source_metadata,
    parse_tool_output,
    run_knowledge_base_search,
    run_web_search_tool,
    run_with_retries,
    sources_to_documents,
    validation_error_result,
    WebSearchToolRequest,
)


class AgentToolRequestValidationTests(unittest.TestCase):
    def test_query_is_required(self):
        with self.assertRaises(ValidationError):
            KnowledgeBaseSearchRequest(query="")

    def test_top_k_is_bounded(self):
        with self.assertRaises(ValidationError):
            KnowledgeBaseSearchRequest(query="hello", top_k=0)
        with self.assertRaises(ValidationError):
            KnowledgeBaseSearchRequest(query="hello", top_k=99)

    def test_max_retries_is_bounded(self):
        with self.assertRaises(ValidationError):
            KnowledgeBaseSearchRequest(query="hello", max_retries=4)

    def test_source_types_are_optional_and_normalized(self):
        request = KnowledgeBaseSearchRequest(query="hello", source_types=[" PDF ", "docx"])
        self.assertEqual(request.source_types, ["pdf", "docx"])

    def test_invalid_source_types_are_rejected(self):
        with self.assertRaises(ValidationError):
            KnowledgeBaseSearchRequest(query="hello", source_types=["pptx"])


class AgentToolResultParsingTests(unittest.TestCase):
    def test_parse_dictionary_result(self):
        payload = {
            "status": "success",
            "content": "Evidence found",
            "sources": [
                {
                    "source": "report.pdf",
                    "title": "Report",
                    "content": "Snippet",
                }
            ],
        }
        parsed = parse_tool_output(payload)
        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed.status, "success")
        self.assertEqual(len(parsed.sources), 1)

    def test_parse_json_string_result(self):
        payload = AgentToolResult(
            status="empty",
            content="No relevant documents were found in the knowledge base.",
            sources=[],
        )
        parsed = parse_tool_output(payload.to_tool_content())
        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed.status, "empty")

    def test_plain_text_fallback_returns_none(self):
        self.assertIsNone(parse_tool_output("plain tool output"))

    def test_sources_convert_to_documents(self):
        documents = sources_to_documents(
            [
                ToolSourceMetadata(
                    source="https://example.com",
                    title="Example",
                    source_type="web",
                    content="External evidence",
                )
            ]
        )
        self.assertEqual(documents[0].metadata["source"], "https://example.com")
        self.assertEqual(documents[0].page_content, "External evidence")

    def test_apply_tool_output_merges_structured_sources(self):
        documents = []
        content = apply_tool_output_to_documents(
            documents,
            {
                "status": "success",
                "content": "Evidence found",
                "sources": [{"source": "report.pdf", "content": "Snippet"}],
            },
        )
        self.assertEqual(content, "Evidence found")
        self.assertEqual(len(documents), 1)


class KnowledgeBaseSearchServiceTests(unittest.TestCase):
    def test_build_retrieval_filter_with_source_types(self):
        self.assertEqual(build_retrieval_filter(None), {"chunk_level": "child"})
        self.assertEqual(
            build_retrieval_filter(["pdf"]),
            {"$and": [{"chunk_level": "child"}, {"source_type": "pdf"}]},
        )
        self.assertEqual(
            build_retrieval_filter(["pdf", "docx"]),
            {"$and": [{"chunk_level": "child"}, {"source_type": {"$in": ["pdf", "docx"]}}]},
        )

    @patch("app.services.indexing.get_vectorstore")
    @patch("app.services.indexing.get_parent_chunks_by_ids")
    def test_knowledge_base_search_applies_collection_top_k_and_source_types(
        self,
        mock_get_parents,
        mock_get_vectorstore,
    ):
        child = MagicMock()
        child.page_content = "child content"
        child.metadata = {
            "source": "report.pdf",
            "source_type": "pdf",
            "parent_chunk_id": "parent-1",
            "section_path": "root",
        }
        vectorstore = MagicMock()
        vectorstore.similarity_search_with_relevance_scores.return_value = [(child, 0.91)]
        mock_get_vectorstore.return_value = vectorstore
        mock_get_parents.return_value = []

        request = KnowledgeBaseSearchRequest(
            query="agent tools",
            collection_name="resume_bank",
            source_types=["pdf"],
            top_k=2,
        )
        result = run_knowledge_base_search(request)

        mock_get_vectorstore.assert_called_once_with("resume_bank")
        vectorstore.similarity_search_with_relevance_scores.assert_called_once_with(
            query="agent tools",
            k=2,
            filter={"$and": [{"chunk_level": "child"}, {"source_type": "pdf"}]},
        )
        self.assertEqual(result.status, "success")
        self.assertEqual(len(result.sources), 1)
        self.assertEqual(result.sources[0].source, "report.pdf")

    @patch("app.services.indexing.get_vectorstore")
    @patch("app.services.indexing.get_parent_chunks_by_ids")
    def test_knowledge_base_search_empty_result(self, mock_get_parents, mock_get_vectorstore):
        vectorstore = MagicMock()
        vectorstore.similarity_search_with_relevance_scores.return_value = []
        mock_get_vectorstore.return_value = vectorstore
        mock_get_parents.return_value = []

        result = run_knowledge_base_search(KnowledgeBaseSearchRequest(query="missing topic"))

        self.assertEqual(result.status, "empty")
        self.assertEqual(result.sources, [])

    @patch("app.services.agent_tools._run_knowledge_base_search_once")
    def test_knowledge_base_search_returns_error_after_retries(self, mock_search_once):
        mock_search_once.side_effect = RuntimeError("vector store unavailable")

        result = run_knowledge_base_search(
            KnowledgeBaseSearchRequest(query="agent tools", max_retries=1)
        )

        self.assertEqual(result.status, "error")
        self.assertIn("vector store unavailable", result.error or "")


class WebSearchToolServiceTests(unittest.TestCase):
    @patch("app.services.web_search.run_web_search")
    def test_web_search_success_returns_structured_sources(self, mock_run_web_search):
        from app.services.web_search import web_results_to_documents, WebSearchResult

        web_docs = web_results_to_documents(
            [
                WebSearchResult(
                    title="Agent search",
                    url="https://example.com/agent",
                    snippet="External evidence",
                )
            ],
            query="agent search",
        )
        mock_run_web_search.return_value = ("[1] title=Agent search", web_docs)

        result = run_web_search_tool(WebSearchToolRequest(query="agent search"))

        self.assertEqual(result.status, "success")
        self.assertEqual(len(result.sources), 1)
        self.assertEqual(result.sources[0].source, "https://example.com/agent")

    @patch("app.services.web_search.run_web_search")
    def test_web_search_empty_result(self, mock_run_web_search):
        mock_run_web_search.return_value = ("No web search results were found.", [])

        result = run_web_search_tool(WebSearchToolRequest(query="missing topic"))

        self.assertEqual(result.status, "empty")
        self.assertEqual(result.sources, [])

    @patch("app.services.web_search.run_web_search")
    def test_web_search_provider_failure(self, mock_run_web_search):
        mock_run_web_search.return_value = ("Web search failed: timeout", [])

        result = run_web_search_tool(WebSearchToolRequest(query="agent search"))

        self.assertEqual(result.status, "error")
        self.assertIn("timeout", result.error or "")

    def test_run_with_retries_eventually_succeeds(self):
        attempts = {"count": 0}

        def operation():
            attempts["count"] += 1
            if attempts["count"] < 2:
                raise RuntimeError("temporary")
            return "ok"

        outcome = run_with_retries(operation, max_retries=2, error_message="failed")
        self.assertEqual(outcome, "ok")
        self.assertEqual(attempts["count"], 2)


class AgentToolHelperTests(unittest.TestCase):
    def test_validation_error_result(self):
        result = validation_error_result("query required")
        self.assertEqual(result.status, "error")
        self.assertEqual(result.error, "query required")

    def test_document_to_source_metadata(self):
        from app.services.agent_tools import Document

        metadata = document_to_source_metadata(
            Document(page_content="body", metadata={"source": "report.pdf", "retrieval_score": 0.8})
        )
        self.assertEqual(metadata.source, "report.pdf")
        self.assertEqual(metadata.retrieval_score, 0.8)


if __name__ == "__main__":
    unittest.main()
