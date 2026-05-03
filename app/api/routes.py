"""FastAPI route definitions for the RAG service."""

from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from app.core.config import settings
from app.core.graph import get_rag_graph
from app.models.schemas import (
    CollectionInfo,
    DeleteDocumentRequest,
    DeleteResponse,
    HealthResponse,
    IndexResponse,
    ListCollectionsResponse,
    QueryRequest,
    QueryResponse,
    SourceDocument,
)
from app.services.indexing import (
    delete_document,
    index_file,
    index_text,
    list_collections,
)

router = APIRouter()


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check() -> HealthResponse:
    """Return the service health status."""
    return HealthResponse(
        status="ok",
        details={
            "llm_model": settings.openai_llm_model,
            "embedding_model": settings.openai_embedding_model,
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
    content = await file.read()
    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )
    try:
        num_chunks = index_file(content, file.filename or "upload", collection_name)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    col = collection_name or settings.chroma_collection_name
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
    if not text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Text content is empty.",
        )
    num_chunks = index_text(text, source=source, collection_name=collection_name)
    col = collection_name or settings.chroma_collection_name
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
    try:
        delete_document(doc_id, collection_name)
    except (ValueError, KeyError) as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document not found: {doc_id}",
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred while deleting the document.",
        ) from exc
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
    graph = get_rag_graph()

    initial_state = {
        "question": body.question,
        "collection_name": body.collection_name,
        "documents": [],
        "answer": "",
    }

    try:
        final_state = await graph.ainvoke(initial_state)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"RAG pipeline error: {exc}",
        ) from exc

    sources = [
        SourceDocument(
            source=doc.metadata.get("source", ""),
            page=doc.metadata.get("page"),
            content=doc.page_content[: settings.source_preview_chars],
        )
        for doc in final_state.get("documents", [])
    ]

    return QueryResponse(
        question=body.question,
        answer=final_state["answer"],
        sources=sources,
    )
