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
    current_query: str
    query_variants: list[str]
    candidate_documents: list[Document]
    documents: list[Document]
    answer: str
    trace: list[str]
    retrieval_hop: int


def _append_trace(state: RAGState, message: str) -> list[str]:
    trace = [*state.get("trace", [])]
    trace.append(f"{len(trace) + 1}. {message}")
    return trace


def _merge_documents(existing: list[Document], incoming: list[Document]) -> list[Document]:
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


def _normalize_rewritten_query(candidate: str, fallback: str) -> str:
    cleaned = candidate.strip()
    if not cleaned or cleaned.lower() in {"none", "n/a", "no additional query", "null"}:
        return fallback
    return cleaned


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


def rewrite_query(state: RAGState) -> RAGState:
    """Rewrite the user question into a retrieval-friendly query for the current hop."""
    next_hop = state.get("retrieval_hop", 0) + 1
    trace = _append_trace(state, f"Preparing retrieval query for hop {next_hop}")
    llm = _get_llm()

    context = "\n\n---\n\n".join(
        doc.page_content[: settings.rewrite_context_chars]
        for doc in state.get("documents", [])
    )

    if state.get("documents"):
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You are a retrieval planner. "
                        "Given the original user question and the relevant context already found, "
                        "write one concise follow-up search query that can fill the remaining knowledge gap. "
                        "Return only the rewritten search query."
                    ),
                ),
                (
                    "human",
                    (
                        "Original question: {question}\n\n"
                        "Relevant context already found:\n{context}\n\n"
                        "Follow-up search query:"
                    ),
                ),
            ]
        )
        rewritten_query = (prompt | llm | StrOutputParser()).invoke(
            {"question": state["question"], "context": context or "(none)"}
        ).strip()
    else:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You are a retrieval query rewriter. "
                        "Rewrite the user's question into one concise search query optimized for document retrieval. "
                        "Keep the meaning unchanged and return only the rewritten query."
                    ),
                ),
                (
                    "human",
                    "User question: {question}\n\nRetrieval query:",
                ),
            ]
        )
        rewritten_query = (prompt | llm | StrOutputParser()).invoke(
            {"question": state["question"]}
        ).strip()

    current_query = _normalize_rewritten_query(rewritten_query, state["question"])
    logger.info("[rag.rewrite] hop=%s query=%s", next_hop, current_query[:200])
    trace.append(f"{len(trace) + 1}. Retrieval query for hop {next_hop}: {current_query}")
    return {
        **state,
        "current_query": current_query,
        "query_variants": [*state.get("query_variants", []), current_query],
        "trace": trace,
    }


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


def retrieve(state: RAGState) -> RAGState:
    """Retrieve the top-k most relevant chunks from the vector store."""
    current_hop = state.get("retrieval_hop", 0) + 1
    trace = _append_trace(state, f"Running retrieval hop {current_hop}")
    logger.info(
        "[rag.retrieve] start collection=%s hop=%s query=%s",
        state.get("collection_name") or settings.chroma_collection_name,
        current_hop,
        state.get("current_query", state["question"])[:120],
    )
    vs = get_vectorstore(state.get("collection_name"))
    scored_docs = vs.similarity_search_with_relevance_scores(
        state.get("current_query") or state["question"],
        k=settings.retriever_top_k,
    )
    docs: list[Document] = []
    scores: list[str] = []
    for doc, score in scored_docs:
        doc.metadata["retrieval_score"] = round(float(score), 4)
        doc.metadata["retrieval_hop"] = current_hop
        doc.metadata["retrieval_query"] = state.get("current_query") or state["question"]
        docs.append(doc)
        scores.append(f"{float(score):.4f}")

    logger.info("[rag.retrieve] complete docs=%s scores=%s", len(docs), scores)
    trace.append(
        f"{len(trace) + 1}. Retrieval hop {current_hop} complete: {len(docs)} candidate chunks found"
    )
    if scores:
        trace.append(f"{len(trace) + 1}. Retrieval hop {current_hop} scores: {', '.join(scores)}")
    return {**state, "candidate_documents": docs, "retrieval_hop": current_hop, "trace": trace}


def grade_docs(state: RAGState) -> RAGState:
    """Filter out documents that are not relevant to the question.

    Uses a lightweight yes/no prompt so that only genuinely related chunks
    are passed to the generation step.
    """
    llm = _get_llm()
    question = state["question"]
    hop_relevant: list[Document] = []
    trace = _append_trace(state, f"Grading retrieval hop {state['retrieval_hop']} for relevance")
    logger.info("[rag.grade] start docs=%s", len(state["candidate_documents"]))

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

    for doc in state["candidate_documents"]:
        messages = grade_prompt.format_messages(
            question=question,
            content=doc.page_content[: settings.grade_context_chars],
        )
        response = llm.invoke(messages)
        verdict = response.content.strip().lower()
        logger.info(
            "[rag.grade] verdict=%s source=%s score=%s",
            verdict,
            doc.metadata.get("source", "unknown"),
            doc.metadata.get("retrieval_score"),
        )
        if verdict.startswith("yes"):
            hop_relevant.append(doc)

    documents = _merge_documents(state.get("documents", []), hop_relevant)
    logger.info("[rag.grade] complete relevant_docs=%s", len(documents))
    trace.append(
        f"{len(trace) + 1}. Hop {state['retrieval_hop']} grading complete: "
        f"{len(hop_relevant)} chunks kept in this hop, {len(documents)} unique chunks total"
    )
    return {**state, "documents": documents, "candidate_documents": [], "trace": trace}


def generate(state: RAGState) -> RAGState:
    """Generate an answer using the graded documents as context."""
    trace = _append_trace(state, "Generating grounded answer from relevant chunks")
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
    trace.append(f"{len(trace) + 1}. Answer generation complete")
    return {**state, "answer": answer, "trace": trace}


def no_answer(state: RAGState) -> RAGState:
    """Return a polite fallback when no relevant documents were found."""
    trace = _append_trace(state, "No relevant chunks remained after grading")
    logger.info("[rag.no_answer] no relevant documents found")
    return {
        **state,
        "answer": (
            "I could not find any relevant information in the knowledge base "
            "to answer your question."
        ),
        "trace": [*trace, f"{len(trace) + 1}. Returned fallback response"],
    }


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def _route_after_grading(state: RAGState) -> str:
    """Decide whether to generate an answer or return a fallback."""
    relevant_docs = len(state["documents"])
    current_hop = state.get("retrieval_hop", 0)

    if relevant_docs >= settings.min_relevant_chunks_to_answer:
        return "generate"
    if current_hop < settings.retrieval_max_hops:
        return "rewrite_query"
    return "generate" if relevant_docs else "no_answer"


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------


def build_rag_graph() -> StateGraph:
    """Compile and return the RAG StateGraph."""
    logger.info("[rag.graph] building graph")
    graph = StateGraph(RAGState)

    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("retrieve", retrieve)
    graph.add_node("grade_docs", grade_docs)
    graph.add_node("generate", generate)
    graph.add_node("no_answer", no_answer)

    graph.add_edge(START, "rewrite_query")
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("retrieve", "grade_docs")
    graph.add_conditional_edges(
        "grade_docs",
        _route_after_grading,
        {
            "rewrite_query": "rewrite_query",
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
