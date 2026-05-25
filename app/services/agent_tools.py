"""Shared Agent tool services for local LangChain tools and MCP-exposed tools."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Literal, TypeVar

from pydantic import BaseModel, Field, ValidationError, field_validator

from app.core.config import settings

logger = logging.getLogger("uvicorn.error")

ToolStatus = Literal["success", "empty", "error"]

MAX_TOP_K = 20
MAX_RETRIES = 3
MAX_TIMEOUT_SECONDS = 60.0
ALLOWED_SOURCE_TYPES = frozenset({"pdf", "docx", "txt", "md", "text", "web"})

try:
    from langchain_core.documents import Document
except ModuleNotFoundError:

    @dataclass
    class Document:  # type: ignore[no-redef]
        page_content: str
        metadata: dict[str, Any]


T = TypeVar("T")


class KnowledgeBaseSearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    collection_name: str | None = None
    source_types: list[str] | None = None
    top_k: int | None = Field(default=None, ge=1, le=MAX_TOP_K)
    timeout_seconds: float | None = Field(default=None, ge=0.1, le=MAX_TIMEOUT_SECONDS)
    max_retries: int = Field(default=0, ge=0, le=MAX_RETRIES)

    @field_validator("source_types")
    @classmethod
    def normalize_source_types(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        normalized = [item.strip().lower() for item in value if str(item).strip()]
        invalid = [item for item in normalized if item not in ALLOWED_SOURCE_TYPES]
        if invalid:
            raise ValueError(
                f"Unsupported source_types: {', '.join(invalid)}. "
                f"Allowed values: {', '.join(sorted(ALLOWED_SOURCE_TYPES))}"
            )
        return normalized or None


class WebSearchToolRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=MAX_TOP_K)
    timeout_seconds: float | None = Field(default=None, ge=0.1, le=MAX_TIMEOUT_SECONDS)
    max_retries: int = Field(default=0, ge=0, le=MAX_RETRIES)
    provider_name: str | None = None


class CollectionOverviewRequest(BaseModel):
    collection_name: str | None = None
    timeout_seconds: float | None = Field(default=None, ge=0.1, le=MAX_TIMEOUT_SECONDS)
    max_retries: int = Field(default=0, ge=0, le=MAX_RETRIES)


class ToolSourceMetadata(BaseModel):
    source: str = ""
    page: int | None = None
    title: str | None = None
    source_type: str | None = None
    chunk_level: str | None = None
    section_path: str | None = None
    parent_chunk_id: str | None = None
    parent_title: str | None = None
    parent_section_path: str | None = None
    parent_content_preview: str | None = None
    content_preview: str | None = None
    retrieval_score: float | None = None
    retrieval_hop: int | None = None
    retrieval_query: str | None = None
    content: str = ""


class AgentToolResult(BaseModel):
    status: ToolStatus
    content: str
    sources: list[ToolSourceMetadata] = Field(default_factory=list)
    error: str | None = None

    def to_documents(self) -> list[Document]:
        documents: list[Document] = []
        for source in self.sources:
            metadata = source.model_dump(exclude={"content"})
            page_content = source.content or source.content_preview or ""
            documents.append(Document(page_content=page_content, metadata=metadata))
        return documents

    def to_tool_payload(self) -> dict[str, Any]:
        return self.model_dump()

    def to_tool_content(self) -> str:
        return json.dumps(self.to_tool_payload(), ensure_ascii=False)


def validation_error_result(message: str) -> AgentToolResult:
    return AgentToolResult(status="error", content=message, error=message)


def run_with_retries(
    operation: Callable[[], T],
    *,
    max_retries: int,
    error_message: str,
) -> T | AgentToolResult:
    attempts = max(0, max_retries) + 1
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            return operation()
        except Exception as exc:
            last_error = exc
            logger.warning(
                "[agent_tools.retry] attempt=%s/%s error=%s",
                attempt + 1,
                attempts,
                exc,
            )
    detail = str(last_error) if last_error else "unknown error"
    message = f"{error_message}: {detail}"
    return AgentToolResult(status="error", content=message, error=message)


def build_retrieval_filter(source_types: list[str] | None) -> dict[str, Any]:
    child_filter: dict[str, Any] = {"chunk_level": "child"}
    if not source_types:
        return child_filter
    if len(source_types) == 1:
        return {"$and": [child_filter, {"source_type": source_types[0]}]}
    return {"$and": [child_filter, {"source_type": {"$in": source_types}}]}


def document_to_source_metadata(document: Document) -> ToolSourceMetadata:
    metadata = document.metadata
    return ToolSourceMetadata(
        source=str(metadata.get("source") or ""),
        page=metadata.get("page"),
        title=metadata.get("title"),
        source_type=metadata.get("source_type"),
        chunk_level=metadata.get("chunk_level"),
        section_path=metadata.get("section_path"),
        parent_chunk_id=metadata.get("parent_chunk_id"),
        parent_title=metadata.get("parent_title"),
        parent_section_path=metadata.get("parent_section_path"),
        parent_content_preview=metadata.get("parent_content_preview"),
        content_preview=metadata.get("content_preview"),
        retrieval_score=metadata.get("retrieval_score"),
        retrieval_hop=metadata.get("retrieval_hop"),
        retrieval_query=metadata.get("retrieval_query"),
        content=document.page_content,
    )


def sources_to_documents(sources: list[ToolSourceMetadata]) -> list[Document]:
    return AgentToolResult(status="success", content="", sources=sources).to_documents()


def parse_tool_output(raw: Any) -> AgentToolResult | None:
    if isinstance(raw, AgentToolResult):
        return raw
    if isinstance(raw, dict):
        try:
            return AgentToolResult.model_validate(raw)
        except ValidationError:
            return None
    if isinstance(raw, str):
        stripped = raw.strip()
        if not stripped:
            return None
        if stripped.startswith("{"):
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError:
                return None
            if isinstance(payload, dict):
                try:
                    return AgentToolResult.model_validate(payload)
                except ValidationError:
                    return None
        return None
    return None


def merge_documents(existing: list[Document], incoming: list[Document]) -> list[Document]:
    merged = list(existing)
    index_by_key = {
        (
            doc.metadata.get("source"),
            doc.metadata.get("page"),
            doc.metadata.get("start_index"),
            doc.page_content,
        ): idx
        for idx, doc in enumerate(merged)
    }

    for doc in incoming:
        key = (
            doc.metadata.get("source"),
            doc.metadata.get("page"),
            doc.metadata.get("start_index"),
            doc.page_content,
        )
        existing_index = index_by_key.get(key)
        if existing_index is None:
            index_by_key[key] = len(merged)
            merged.append(doc)
            continue

        existing_score = merged[existing_index].metadata.get("retrieval_score")
        incoming_score = doc.metadata.get("retrieval_score")
        if incoming_score is not None and (
            existing_score is None or float(incoming_score) > float(existing_score)
        ):
            merged[existing_index] = doc

    return merged


def apply_tool_output_to_documents(documents: list[Document], raw: Any) -> str:
    parsed = parse_tool_output(raw)
    if parsed is None:
        return _stringify_content(raw)

    if parsed.sources:
        merged = merge_documents(documents, parsed.to_documents())
        documents.clear()
        documents.extend(merged)
    return parsed.content or _stringify_content(raw)


def _stringify_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content") or ""
                if text:
                    parts.append(str(text))
        return "\n".join(part for part in parts if part).strip()
    if content is None:
        return ""
    return str(content)


def _format_parent_child_documents(
    child_docs: list[Document],
    parent_docs: list[Document],
) -> str:
    if not child_docs:
        return "No relevant documents were found in the knowledge base."

    parent_by_id = {
        str(doc.metadata.get("parent_chunk_id") or ""): doc
        for doc in parent_docs
        if doc.metadata.get("parent_chunk_id")
    }
    grouped_children: dict[str, list[Document]] = {}
    for child_doc in child_docs:
        parent_id = str(child_doc.metadata.get("parent_chunk_id") or "")
        grouped_children.setdefault(parent_id, []).append(child_doc)

    parts: list[str] = []
    for group_index, child_doc in enumerate(child_docs, start=1):
        parent_id = str(child_doc.metadata.get("parent_chunk_id") or "")
        if group_index > 1 and parent_id in grouped_children:
            if grouped_children[parent_id][0] is not child_doc:
                continue

        source = child_doc.metadata.get("source", "unknown")
        section_path = child_doc.metadata.get("section_path") or "root"
        retrieval_score = child_doc.metadata.get("retrieval_score")
        score_text = f"{float(retrieval_score):.4f}" if retrieval_score is not None else "n/a"
        matched_children = grouped_children.get(parent_id, [child_doc])
        parent_doc = parent_by_id.get(parent_id)

        header = f"[{group_index}] source={source}; section={section_path}; score={score_text}"
        if parent_doc is not None:
            parent_title = parent_doc.metadata.get("title") or parent_doc.metadata.get("section_path") or "root"
            parent_preview = (
                parent_doc.metadata.get("content_preview") or parent_doc.page_content
            )[: settings.rewrite_context_chars]
            child_parts = []
            for child_index, matched_child in enumerate(matched_children, start=1):
                child_preview = (
                    matched_child.metadata.get("content_preview") or matched_child.page_content
                )[: settings.source_preview_chars]
                child_parts.append(f"  Child {child_index}: {child_preview}")
            parts.append(
                f"{header}\nParent context [{parent_title}]:\n{parent_preview}\nMatched child evidence:\n"
                + "\n".join(child_parts)
            )
        else:
            child_preview = (
                child_doc.metadata.get("content_preview") or child_doc.page_content
            )[: settings.source_preview_chars]
            parts.append(f"{header}\n{child_preview}")
    return "\n\n".join(parts)


def _build_retrieval_audit_text(
    child_docs: list[Document],
    parent_docs: list[Document],
) -> str:
    if not child_docs:
        return "Retrieval audit: no child chunks matched the query."

    child_lines: list[str] = []
    for index, child_doc in enumerate(child_docs, start=1):
        child_lines.append(
            "- "
            f"child[{index}] source={child_doc.metadata.get('source', 'unknown')}; "
            f"title={child_doc.metadata.get('title') or 'n/a'}; "
            f"section={child_doc.metadata.get('section_path') or 'root'}; "
            f"parent_chunk_id={child_doc.metadata.get('parent_chunk_id') or 'n/a'}; "
            f"score={child_doc.metadata.get('retrieval_score', 'n/a')}"
        )

    parent_lines: list[str] = []
    for index, parent_doc in enumerate(parent_docs, start=1):
        preview = (parent_doc.metadata.get("content_preview") or parent_doc.page_content)[:200]
        parent_lines.append(
            "- "
            f"parent[{index}] title={parent_doc.metadata.get('title') or 'n/a'}; "
            f"section={parent_doc.metadata.get('section_path') or parent_doc.metadata.get('parent_section_path') or 'root'}; "
            f"parent_chunk_id={parent_doc.metadata.get('parent_chunk_id') or 'n/a'}; "
            f"preview={preview}"
        )

    parts = ["Retrieval audit", "Matched child chunks:", *child_lines]
    if parent_lines:
        parts.extend(["Recovered parent contexts:", *parent_lines])
    else:
        parts.append("Recovered parent contexts: none")
    return "\n".join(parts)


def _run_knowledge_base_search_once(request: KnowledgeBaseSearchRequest) -> AgentToolResult:
    from app.services.indexing import get_parent_chunks_by_ids, get_vectorstore

    resolved_collection = request.collection_name or settings.chroma_collection_name
    top_k = request.top_k or settings.retriever_top_k
    vectorstore = get_vectorstore(request.collection_name)
    scored_docs = vectorstore.similarity_search_with_relevance_scores(
        query=request.query,
        k=top_k,
        filter=build_retrieval_filter(request.source_types),
    )

    tool_docs: list[Document] = []
    parent_chunk_ids: list[str] = []
    for doc, score in scored_docs:
        metadata = dict(doc.metadata)
        metadata["retrieval_score"] = round(float(score), 4)
        metadata["retrieval_hop"] = 1
        metadata["retrieval_query"] = request.query
        tool_docs.append(Document(page_content=doc.page_content, metadata=metadata))
        parent_chunk_id = str(metadata.get("parent_chunk_id") or "").strip()
        if parent_chunk_id:
            parent_chunk_ids.append(parent_chunk_id)

    parent_docs = get_parent_chunks_by_ids(parent_chunk_ids, request.collection_name)
    parent_by_id = {
        str(doc.metadata.get("parent_chunk_id") or ""): doc
        for doc in parent_docs
        if doc.metadata.get("parent_chunk_id")
    }
    grouped_children: dict[str, list[Document]] = {}
    for tool_doc in tool_docs:
        parent_chunk_id = str(tool_doc.metadata.get("parent_chunk_id") or "")
        grouped_children.setdefault(parent_chunk_id, []).append(tool_doc)

    enriched_tool_docs: list[Document] = []
    for tool_doc in tool_docs:
        metadata = dict(tool_doc.metadata)
        parent_chunk_id = str(metadata.get("parent_chunk_id") or "")
        parent_doc = parent_by_id.get(parent_chunk_id)
        if parent_doc is not None:
            metadata["parent_title"] = parent_doc.metadata.get("title") or parent_doc.metadata.get("section_path")
            metadata["parent_section_path"] = parent_doc.metadata.get("section_path") or parent_doc.metadata.get(
                "parent_section_path"
            )
            metadata["parent_content_preview"] = (
                parent_doc.metadata.get("content_preview") or parent_doc.page_content
            )[: settings.rewrite_context_chars]
            metadata["matched_child_count"] = len(grouped_children.get(parent_chunk_id, []))
        enriched_tool_docs.append(Document(page_content=tool_doc.page_content, metadata=metadata))

    if not enriched_tool_docs:
        message = "No relevant documents were found in the knowledge base."
        return AgentToolResult(status="empty", content=message, sources=[])

    retrieval_audit = _build_retrieval_audit_text(enriched_tool_docs, parent_docs)
    formatted = _format_parent_child_documents(enriched_tool_docs, parent_docs)
    content = retrieval_audit + "\n\n" + formatted
    sources = [document_to_source_metadata(doc) for doc in enriched_tool_docs]
    logger.info(
        "[agent_tools.kb_search] collection=%s results=%s parents=%s source_types=%s",
        resolved_collection,
        len(enriched_tool_docs),
        len(parent_docs),
        request.source_types,
    )
    return AgentToolResult(status="success", content=content, sources=sources)


def run_knowledge_base_search(request: KnowledgeBaseSearchRequest) -> AgentToolResult:
    timeout = request.timeout_seconds or settings.agent_tool_timeout_seconds

    def operation() -> AgentToolResult:
        start = time.monotonic()
        result = _run_knowledge_base_search_once(request)
        elapsed = time.monotonic() - start
        if elapsed > timeout:
            raise TimeoutError(f"Knowledge base search exceeded timeout of {timeout}s")
        return result

    outcome = run_with_retries(
        operation,
        max_retries=request.max_retries,
        error_message="Knowledge base search failed",
    )
    if isinstance(outcome, AgentToolResult):
        return outcome
    return outcome


def run_collection_overview(request: CollectionOverviewRequest) -> AgentToolResult:
    from app.services.indexing import list_collections

    resolved_collection = request.collection_name or settings.chroma_collection_name

    def operation() -> AgentToolResult:
        collections = list_collections()
        summary = ", ".join(f"{item['name']}({item['count']})" for item in collections[:10]) or "none"
        content = (
            f"Active LLM model: {settings.llm_model}\n"
            f"Active embedding model: {settings.embedding_model}\n"
            f"Default collection: {settings.chroma_collection_name}\n"
            f"Requested collection: {resolved_collection}\n"
            f"Collections: {summary}"
        )
        return AgentToolResult(status="success", content=content, sources=[])

    outcome = run_with_retries(
        operation,
        max_retries=request.max_retries,
        error_message="Collection overview failed",
    )
    if isinstance(outcome, AgentToolResult):
        return outcome
    return outcome


def run_web_search_tool(request: WebSearchToolRequest) -> AgentToolResult:
    from app.services.web_search import run_web_search, web_results_to_documents

    def operation() -> AgentToolResult:
        output, web_docs = run_web_search(request.query, provider_name=request.provider_name)
        if not web_docs:
            if output.startswith("Web search failed"):
                return AgentToolResult(status="error", content=output, error=output, sources=[])
            return AgentToolResult(status="empty", content=output, sources=[])

        sources = [document_to_source_metadata(doc) for doc in web_docs]
        return AgentToolResult(status="success", content=output, sources=sources)

    outcome = run_with_retries(
        operation,
        max_retries=request.max_retries,
        error_message="Web search failed",
    )
    if isinstance(outcome, AgentToolResult):
        return outcome
    return outcome
