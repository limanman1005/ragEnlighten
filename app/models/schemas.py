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


class ChatHistoryMessage(BaseModel):
    """A single message used to provide chat history to the React Agent endpoint."""

    role: str = Field(
        default="user",
        description="Message role: user, assistant, or system",
    )
    content: str = Field(..., min_length=1, description="Message content")
    reasoning_content: str | None = Field(
        default=None,
        description="Optional hidden reasoning content returned by reasoning models and required for follow-up turns",
    )


class ReactAgentQueryRequest(QueryRequest):
    """Request body for the React Agent chat endpoint."""

    history: list[ChatHistoryMessage] = Field(
        default_factory=list,
        description="Optional prior chat messages for multi-turn conversations",
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
    section_path: str | None = Field(default=None, description="Hierarchical section path for this chunk")
    parent_chunk_id: str | None = Field(default=None, description="Parent chunk identifier for this child chunk")
    retrieval_score: float | None = Field(
        default=None,
        description="Similarity score returned by the vector retrieval stage",
    )
    retrieval_hop: int | None = Field(
        default=None,
        description="The retrieval hop that surfaced this chunk",
    )
    content: str = Field(description="Text excerpt of the retrieved chunk")


class ToolCall(BaseModel):
    """A single tool invocation recorded during the agent workflow."""

    name: str
    status: str
    input_summary: str = ""
    output_summary: str = ""


class ValidationReport(BaseModel):
    """Validation outcome for the generated answer."""

    passed: bool
    confidence: float | None = None
    citations_verified: bool = False
    issues: list[str] = Field(default_factory=list)


class DebugEvent(BaseModel):
    """A single agent debug event for tracing internal execution phases."""

    phase: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    """Response body for the /query endpoint."""

    question: str
    answer: str
    reasoning_content: str | None = None
    question_type: str = "document_qa"
    route: str = "rag"
    plan: list[str] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    sources: list[SourceDocument] = Field(default_factory=list)
    trace: list[str] = Field(default_factory=list)
    debug_events: list[DebugEvent] = Field(default_factory=list)
    confidence_score: float | None = None
    needs_human_review: bool = False
    human_review_reason: str | None = None
    validation: ValidationReport | None = None


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
