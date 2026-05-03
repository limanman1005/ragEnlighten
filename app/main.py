"""FastAPI application entry point."""

from __future__ import annotations

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core.config import settings

app = FastAPI(
    title="ragEnlighten",
    description=(
        "Agentic RAG service powered by LangChain, LangGraph and FastAPI.\n\n"
        "**Workflow**\n"
        "1. Upload documents via `/documents/upload` or index text via `/documents/text`.\n"
        "2. Ask questions via `/query` — the LangGraph pipeline retrieves, grades and "
        "synthesises an answer from your knowledge base.\n"
        "3. Ask questions via `/chat/react-agent` — a ReAct-style agent decides when to call "
        "retrieval or metadata tools before producing a grounded answer.\n"
        "4. Stream React Agent output via `/chat/react-agent/stream` as NDJSON events for progressive UI updates."
    ),
    version="1.0.0",
)

# Allow all origins in development; tighten in production as needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


@app.get("/", tags=["System"])
async def root() -> dict[str, str]:
    """Redirect hint — visit /docs for the interactive API documentation."""
    return {"message": "Welcome to ragEnlighten! Visit /docs for API documentation."}


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.app_reload,
    )
