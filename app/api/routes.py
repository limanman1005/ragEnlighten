"""FastAPI route definitions for the RAG service."""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse

from app.core.config import settings
from app.core.graph import get_rag_graph
from app.models.schemas import (
    CollectionInfo,
    DebugEvent,
    DeleteDocumentRequest,
    DeleteResponse,
    HealthResponse,
    IndexResponse,
    ListCollectionsResponse,
    ReactAgentQueryRequest,
    QueryRequest,
    QueryResponse,
    SourceDocument,
    ToolCall,
    ValidationReport,
)
from app.services.indexing import (
    delete_document,
    index_file,
    index_text,
    list_collections,
)
from app.services.react_agent import run_react_agent_query, stream_react_agent_query

router = APIRouter()
logger = logging.getLogger("uvicorn.error")


def _build_source_documents(documents: list) -> list[SourceDocument]:
    return [
        SourceDocument(
            source=doc.metadata.get("source", ""),
            page=doc.metadata.get("page"),
            section_path=doc.metadata.get("section_path"),
            parent_chunk_id=doc.metadata.get("parent_chunk_id"),
            retrieval_score=doc.metadata.get("retrieval_score"),
            retrieval_hop=doc.metadata.get("retrieval_hop"),
            content=(doc.metadata.get("content_preview") or doc.page_content)[: settings.source_preview_chars],
        )
        for doc in documents
    ]


def _build_query_response(question: str, result: dict) -> QueryResponse:
    sources = _build_source_documents(result.get("documents", []))
    validation = result.get("validation") or {}
    return QueryResponse(
        question=question,
        answer=result.get("answer", ""),
        reasoning_content=result.get("reasoning_content"),
        question_type=result.get("question_type", "document_qa"),
        route=result.get("route", "react_agent"),
        plan=result.get("plan", []),
        tool_calls=[ToolCall(**call) for call in result.get("tool_calls", [])],
        sources=sources,
        trace=result.get("trace", []),
        debug_events=[DebugEvent(**event) for event in result.get("debug_events", [])],
        confidence_score=result.get("confidence_score"),
        needs_human_review=result.get("needs_human_review", False),
        human_review_reason=result.get("human_review_reason"),
        validation=ValidationReport(
            passed=validation.get("passed", False),
            confidence=validation.get("confidence"),
            citations_verified=validation.get("citations_verified", False),
            issues=validation.get("issues", []),
        ),
    )


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check() -> HealthResponse:
    """Return the service health status."""
    return HealthResponse(
        status="ok",
        details={
            "llm_model": settings.llm_model,
            "embedding_model": settings.embedding_model,
            "vector_store": settings.chroma_persist_dir,
        },
    )


# ---------------------------------------------------------------------------
# Document ingestion
# ---------------------------------------------------------------------------


@router.post(
    "/documents/upload",
    response_model=IndexResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Documents"],
)
async def upload_document(
    file: UploadFile = File(..., description="Document file to index (.pdf, .docx, .txt, .md)"),
    collection_name: str | None = Form(default=None, description="Target collection name"),
) -> IndexResponse:
    """Upload a file and index its content into the vector store."""
    logger.info(
        "[documents.upload] start filename=%s collection=%s",
        file.filename,
        collection_name or settings.chroma_collection_name,
    )
    content = await file.read()
    if not content:
        logger.warning("[documents.upload] rejected empty file filename=%s", file.filename)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )
    try:
        num_chunks = index_file(content, file.filename or "upload", collection_name)
    except ValueError as exc:
        logger.exception("[documents.upload] indexing failed filename=%s", file.filename)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    col = collection_name or settings.chroma_collection_name
    logger.info(
        "[documents.upload] complete filename=%s collection=%s chunks=%s",
        file.filename,
        col,
        num_chunks,
    )
    return IndexResponse(
        message=f"File '{file.filename}' indexed successfully.",
        num_chunks=num_chunks,
        collection_name=col,
    )


@router.post(
    "/documents/text",
    response_model=IndexResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Documents"],
)
async def index_text_document(
    text: str = Form(..., description="Raw text to index"),
    source: str = Form(default="inline", description="Label / source identifier for the text"),
    collection_name: str | None = Form(default=None, description="Target collection name"),
) -> IndexResponse:
    """Index a plain-text snippet directly (no file upload required)."""
    logger.info(
        "[documents.text] start source=%s collection=%s text_chars=%s",
        source,
        collection_name or settings.chroma_collection_name,
        len(text),
    )
    if not text.strip():
        logger.warning("[documents.text] rejected empty text source=%s", source)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Text content is empty.",
        )
    num_chunks = index_text(text, source=source, collection_name=collection_name)
    col = collection_name or settings.chroma_collection_name
    logger.info(
        "[documents.text] complete source=%s collection=%s chunks=%s",
        source,
        col,
        num_chunks,
    )
    return IndexResponse(
        message="Text indexed successfully.",
        num_chunks=num_chunks,
        collection_name=col,
    )


# ---------------------------------------------------------------------------
# Document deletion
# ---------------------------------------------------------------------------


@router.delete(
    "/documents/{doc_id}",
    response_model=DeleteResponse,
    tags=["Documents"],
)
async def remove_document(
    doc_id: str,
    body: DeleteDocumentRequest | None = None,
) -> DeleteResponse:
    """Delete a document chunk by its vector-store ID."""
    collection_name = body.collection_name if body else None
    logger.info(
        "[documents.delete] start doc_id=%s collection=%s",
        doc_id,
        collection_name or settings.chroma_collection_name,
    )
    try:
        delete_document(doc_id, collection_name)
    except (ValueError, KeyError) as exc:
        logger.exception("[documents.delete] document not found doc_id=%s", doc_id)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document not found: {doc_id}",
        ) from exc
    except RuntimeError as exc:
        logger.exception("[documents.delete] delete failed doc_id=%s", doc_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred while deleting the document.",
        ) from exc
    logger.info("[documents.delete] complete doc_id=%s", doc_id)
    return DeleteResponse(message="Document deleted successfully.", doc_id=doc_id)


# ---------------------------------------------------------------------------
# Collections
# ---------------------------------------------------------------------------


@router.get(
    "/collections",
    response_model=ListCollectionsResponse,
    tags=["Documents"],
)
async def list_all_collections() -> ListCollectionsResponse:
    """List all vector-store collections and their document counts."""
    cols = list_collections()
    return ListCollectionsResponse(
        collections=[CollectionInfo(name=c["name"], count=c["count"]) for c in cols]
    )


# ---------------------------------------------------------------------------
# Query / RAG
# ---------------------------------------------------------------------------


@router.post(
    "/query",
    response_model=QueryResponse,
    tags=["RAG"],
)
async def query(body: QueryRequest) -> QueryResponse:
    """Answer a question using the RAG pipeline.

    The request is routed through the LangGraph workflow:
    retrieve → grade → generate (or fallback if no relevant docs found).
    """
    logger.info(
        "[rag.query] start collection=%s question=%s",
        body.collection_name or settings.chroma_collection_name,
        body.question[:120],
    )
    graph = get_rag_graph()

    initial_state = {
        "question": body.question,
        "collection_name": body.collection_name,
        "current_query": body.question,
        "query_variants": [],
        "candidate_documents": [],
        "documents": [],
        "answer": "",
        "trace": ["1. Query accepted by API"],
        "retrieval_hop": 0,
        "question_type": "document_qa",
        "route": "rag",
        "risk_level": "low",
        "plan_steps": [],
        "planned_tools": [],
        "tool_calls": [],
        "tool_context": [],
        "confidence_score": None,
        "validation_passed": False,
        "validation_issues": [],
        "citations_verified": False,
        "retry_count": 0,
        "needs_human_review": False,
        "human_review_reason": "",
    }

    try:
        final_state = await graph.ainvoke(initial_state)
    except Exception as exc:
        logger.exception("[rag.query] graph failed collection=%s", body.collection_name)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"RAG pipeline error: {exc}",
        ) from exc

    sources = _build_source_documents(final_state.get("documents", []))

    response = QueryResponse(
        question=body.question,
        answer=final_state["answer"],
        reasoning_content=final_state.get("reasoning_content"),
        question_type=final_state.get("question_type", "document_qa"),
        route=final_state.get("route", "rag"),
        plan=final_state.get("plan_steps", []),
        tool_calls=[ToolCall(**call) for call in final_state.get("tool_calls", [])],
        sources=sources,
        trace=final_state.get("trace", []),
        debug_events=[DebugEvent(**event) for event in final_state.get("debug_events", [])],
        confidence_score=final_state.get("confidence_score"),
        needs_human_review=final_state.get("needs_human_review", False),
        human_review_reason=final_state.get("human_review_reason") or None,
        validation=ValidationReport(
            passed=final_state.get("validation_passed", False),
            confidence=final_state.get("confidence_score"),
            citations_verified=final_state.get("citations_verified", False),
            issues=final_state.get("validation_issues", []),
        ),
    )
    logger.info(
        "[rag.query] complete collection=%s sources=%s answer_chars=%s",
        body.collection_name or settings.chroma_collection_name,
        len(sources),
        len(response.answer),
    )
    return response


@router.post(
    "/chat/react-agent",
    response_model=QueryResponse,
    tags=["Chat"],
)
async def chat_with_react_agent(body: ReactAgentQueryRequest) -> QueryResponse:
    """Answer a question using a ReAct-style agent loop with retrieval tools."""
    logger.info(
        "[react_agent.chat] start collection=%s question=%s history=%s",
        body.collection_name or settings.chroma_collection_name,
        body.question[:120],
        len(body.history),
    )

    try:
        result = await run_react_agent_query(
            question=body.question,
            collection_name=body.collection_name,
            history=[item.model_dump() for item in body.history],
        )
    except Exception as exc:
        logger.exception("[react_agent.chat] agent failed collection=%s", body.collection_name)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"React Agent pipeline error: {exc}",
        ) from exc

    response = _build_query_response(body.question, result)
    logger.info(
        "[react_agent.chat] complete collection=%s sources=%s answer_chars=%s",
        body.collection_name or settings.chroma_collection_name,
        len(response.sources),
        len(response.answer),
    )
    return response


@router.post(
    "/chat/react-agent/stream",
    tags=["Chat"],
)
async def stream_chat_with_react_agent(body: ReactAgentQueryRequest) -> StreamingResponse:
    """Stream React Agent progress and final answer as NDJSON events."""

    async def event_generator():
        try:
            async for event in stream_react_agent_query(
                question=body.question,
                collection_name=body.collection_name,
                history=[item.model_dump() for item in body.history],
            ):
                if event.get("type") == "final":
                    response = _build_query_response(body.question, event.get("data", {}))
                    payload = {"type": "final", "data": response.model_dump(mode="json")}
                else:
                    payload = event
                yield json.dumps(payload, ensure_ascii=False) + "\n"
        except Exception as exc:
            logger.exception("[react_agent.stream] agent failed collection=%s", body.collection_name)
            yield json.dumps(
                {"type": "error", "data": f"React Agent stream error: {exc}"},
                ensure_ascii=False,
            ) + "\n"

    logger.info(
        "[react_agent.stream] start collection=%s question=%s history=%s",
        body.collection_name or settings.chroma_collection_name,
        body.question[:120],
        len(body.history),
    )
    return StreamingResponse(event_generator(), media_type="application/x-ndjson")
