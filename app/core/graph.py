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

from typing import Annotated, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from app.core.config import settings
from app.services.indexing import get_vectorstore


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
    vs = get_vectorstore(state.get("collection_name"))
    retriever = vs.as_retriever(search_kwargs={"k": settings.retriever_top_k})
    docs = retriever.invoke(state["question"])
    return {**state, "documents": docs}


def grade_docs(state: RAGState) -> RAGState:
    """Filter out documents that are not relevant to the question.

    Uses a lightweight yes/no prompt so that only genuinely related chunks
    are passed to the generation step.
    """
    llm = _get_llm()
    question = state["question"]
    relevant: list[Document] = []

    grade_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content=(
                    "You are a relevance grader. "
                    "Given a user question and a document excerpt, "
                    "reply with a single word: 'yes' if the document is relevant "
                    "to answering the question, or 'no' if it is not."
                )
            ),
            HumanMessage(
                content=(
                    "Question: {question}\n\n"
                    "Document excerpt:\n{content}\n\n"
                    "Is this document relevant? (yes/no)"
                )
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
        if verdict.startswith("yes"):
            relevant.append(doc)

    return {**state, "documents": relevant}


def generate(state: RAGState) -> RAGState:
    """Generate an answer using the graded documents as context."""
    llm = _get_llm()
    context = "\n\n---\n\n".join(doc.page_content for doc in state["documents"])

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content=(
                    "You are a helpful assistant. "
                    "Answer the user's question based solely on the provided context. "
                    "If the context does not contain enough information, say so honestly."
                )
            ),
            HumanMessage(
                content=(
                    "Context:\n{context}\n\n"
                    "Question: {question}\n\n"
                    "Answer:"
                )
            ),
        ]
    )

    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": state["question"]})
    return {**state, "answer": answer}


def no_answer(state: RAGState) -> RAGState:
    """Return a polite fallback when no relevant documents were found."""
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

    return graph.compile()


# Thread-safe lazy singleton for the compiled graph
import threading as _threading

_graph = None
_graph_lock = _threading.Lock()


def get_rag_graph():
    global _graph
    if _graph is None:
        with _graph_lock:
            if _graph is None:
                _graph = build_rag_graph()
    return _graph
