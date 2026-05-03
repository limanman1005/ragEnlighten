"""LangGraph-based Agentic RAG workflow.

Graph topology
--------------

    START
      │
      ▼
  [retrieve]  ──── retrieve relevant documents from the vector store
      │
      ▼
  [grade_docs]  ── keep only documents whose content is relevant to the query
      │
      ├─ no relevant docs ──▶ [no_answer]  ──▶ END
      │
      ▼
  [generate]  ──── synthesise an answer from the graded documents
      │
      ▼
     END
"""

from __future__ import annotations

import logging
from typing import TypedDict

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from app.core.config import settings
from app.services.indexing import get_vectorstore


logger = logging.getLogger("uvicorn.error")


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------


class RAGState(TypedDict):
    """Shared state that flows through every node of the graph."""

    question: str
    collection_name: str | None
    documents: list[Document]
    answer: str


# ---------------------------------------------------------------------------
# Node helpers
# ---------------------------------------------------------------------------


def _get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.llm_model,
        temperature=0,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
    )


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


def retrieve(state: RAGState) -> RAGState:
    """Retrieve the top-k most relevant chunks from the vector store."""
    logger.info(
        "[rag.retrieve] start collection=%s question=%s",
        state.get("collection_name") or settings.chroma_collection_name,
        state["question"][:120],
    )
    vs = get_vectorstore(state.get("collection_name"))
    retriever = vs.as_retriever(search_kwargs={"k": settings.retriever_top_k})
    docs = retriever.invoke(state["question"])
    logger.info("[rag.retrieve] complete docs=%s", len(docs))
    return {**state, "documents": docs}


def grade_docs(state: RAGState) -> RAGState:
    """Filter out documents that are not relevant to the question.

    Uses a lightweight yes/no prompt so that only genuinely related chunks
    are passed to the generation step.
    """
    llm = _get_llm()
    question = state["question"]
    relevant: list[Document] = []
    logger.info("[rag.grade] start docs=%s", len(state["documents"]))

    grade_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are a relevance grader. "
                    "Given a user question and a document excerpt, "
                    "reply with a single word: 'yes' if the document is relevant "
                    "to answering the question, or 'no' if it is not."
                ),
            ),
            (
                "human",
                (
                    "Question: {question}\n\n"
                    "Document excerpt:\n{content}\n\n"
                    "Is this document relevant? (yes/no)"
                ),
            ),
        ]
    )

    for doc in state["documents"]:
        messages = grade_prompt.format_messages(
            question=question,
            content=doc.page_content[: settings.grade_context_chars],
        )
        response = llm.invoke(messages)
        verdict = response.content.strip().lower()
        logger.info(
            "[rag.grade] verdict=%s source=%s",
            verdict,
            doc.metadata.get("source", "unknown"),
        )
        if verdict.startswith("yes"):
            relevant.append(doc)

    logger.info("[rag.grade] complete relevant_docs=%s", len(relevant))
    return {**state, "documents": relevant}


def generate(state: RAGState) -> RAGState:
    """Generate an answer using the graded documents as context."""
    logger.info("[rag.generate] start docs=%s", len(state["documents"]))
    llm = _get_llm()
    context = "\n\n---\n\n".join(doc.page_content for doc in state["documents"])

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are a helpful assistant. "
                    "Answer the user's question based solely on the provided context. "
                    "If the context does not contain enough information, say so honestly."
                ),
            ),
            (
                "human",
                (
                    "Context:\n{context}\n\n"
                    "Question: {question}\n\n"
                    "Answer:"
                ),
            ),
        ]
    )

    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": state["question"]})
    logger.info("[rag.generate] complete answer_chars=%s", len(answer))
    return {**state, "answer": answer}


def no_answer(state: RAGState) -> RAGState:
    """Return a polite fallback when no relevant documents were found."""
    logger.info("[rag.no_answer] no relevant documents found")
    return {
        **state,
        "answer": (
            "I could not find any relevant information in the knowledge base "
            "to answer your question."
        ),
    }


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def _route_after_grading(state: RAGState) -> str:
    """Decide whether to generate an answer or return a fallback."""
    return "generate" if state["documents"] else "no_answer"


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------


def build_rag_graph() -> StateGraph:
    """Compile and return the RAG StateGraph."""
    logger.info("[rag.graph] building graph")
    graph = StateGraph(RAGState)

    graph.add_node("retrieve", retrieve)
    graph.add_node("grade_docs", grade_docs)
    graph.add_node("generate", generate)
    graph.add_node("no_answer", no_answer)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "grade_docs")
    graph.add_conditional_edges(
        "grade_docs",
        _route_after_grading,
        {
            "generate": "generate",
            "no_answer": "no_answer",
        },
    )
    graph.add_edge("generate", END)
    graph.add_edge("no_answer", END)

    compiled = graph.compile()
    logger.info("[rag.graph] build complete")
    return compiled


# Thread-safe lazy singleton for the compiled graph
import threading as _threading

_graph = None
_graph_lock = _threading.Lock()


def get_rag_graph():
    global _graph
    if _graph is None:
        with _graph_lock:
            if _graph is None:
                logger.info("[rag.graph] initializing singleton graph")
                _graph = build_rag_graph()
    return _graph
