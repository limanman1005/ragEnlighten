from typing import Any
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    """Request body for the /query endpoint."""

    question: str = Field(..., min_length=1, description="The question to answer")
    collection_name: str | None = Field(
        default=None,
        description="Optional vector-store collection to query (defaults to the global collection)",
    )


class DeleteDocumentRequest(BaseModel):
    """Request body for the /documents/{doc_id} DELETE endpoint."""

    collection_name: str | None = Field(
        default=None,
        description="Optional collection name; defaults to the global collection",
    )


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------


class SourceDocument(BaseModel):
    """Metadata and excerpt for a single retrieved document chunk."""

    source: str = Field(default="", description="Source file or URL")
    page: int | None = Field(default=None, description="Page number (if available)")
    content: str = Field(description="Text excerpt of the retrieved chunk")


class QueryResponse(BaseModel):
    """Response body for the /query endpoint."""

    question: str
    answer: str
    sources: list[SourceDocument] = Field(default_factory=list)
    trace: list[str] = Field(default_factory=list)


class IndexResponse(BaseModel):
    """Response body for document ingestion endpoints."""

    message: str
    num_chunks: int = Field(description="Number of chunks stored in the vector store")
    collection_name: str


class DeleteResponse(BaseModel):
    """Response body for document deletion."""

    message: str
    doc_id: str


class CollectionInfo(BaseModel):
    """Information about a single collection."""

    name: str
    count: int = Field(description="Number of document chunks in the collection")


class ListCollectionsResponse(BaseModel):
    """Response body for listing all collections."""

    collections: list[CollectionInfo]


class HealthResponse(BaseModel):
    """Health-check response."""

    status: str
    details: dict[str, Any] = Field(default_factory=dict)
