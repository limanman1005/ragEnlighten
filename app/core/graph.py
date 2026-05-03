"""LangGraph-based Agentic RAG workflow.

Graph topology
--------------

    START
      │
      ▼
  [classify_question] ── classify task type, route, and risk level
      │
      ▼
  [plan_question] ───── plan steps and candidate tools
      │
      ├─ route=internal_api ──▶ [call_internal_api] ──▶ [generate]
      │
      ├─ route=rag ──────────▶ [rewrite_query] ──▶ [retrieve] ──▶ [grade_docs]
      │                                              │
      │                                              ├─ enough relevant docs ──▶ [generate]
      │                                              ├─ more hops allowed ────▶ [rewrite_query]
      │                                              └─ no relevant docs ─────▶ [no_answer] ──▶ END
      │
      └─ route=direct ───────▶ [generate]
      │
      ▼
  [validate_answer] ── check confidence and grounding
      │
      ├─ validation passed ──▶ [finalize] ──▶ END
      ├─ low confidence ─────▶ [reflect_and_retry] ──▶ [rewrite_query]
      └─ human review needed ▶ [human_review] ──▶ END
"""

from __future__ import annotations

import json
import logging
import re
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
    question_type: str
    route: str
    risk_level: str
    plan_steps: list[str]
    planned_tools: list[str]
    tool_calls: list[dict[str, str]]
    tool_context: list[str]
    confidence_score: float | None
    validation_passed: bool
    validation_issues: list[str]
    citations_verified: bool
    retry_count: int
    needs_human_review: bool
    human_review_reason: str


def _append_trace(state: RAGState, message: str) -> list[str]:
    trace = [*state.get("trace", [])]
    trace.append(f"{len(trace) + 1}. {message}")
    return trace


def _append_tool_call(
    state: RAGState,
    name: str,
    status: str,
    input_summary: str,
    output_summary: str,
) -> list[dict[str, str]]:
    tool_calls = [*state.get("tool_calls", [])]
    tool_calls.append(
        {
            "name": name,
            "status": status,
            "input_summary": input_summary,
            "output_summary": output_summary,
        }
    )
    return tool_calls


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


def _extract_json_object(text: str) -> dict[str, object]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in model response")
    return json.loads(match.group(0))


def _heuristic_classification(question: str) -> dict[str, object]:
    lower = question.lower()
    internal_keywords = ["collection", "collections", "health", "status", "model", "embedding", "配置", "集合", "模型"]
    high_risk_keywords = [
        "付款",
        "转账",
        "合同",
        "法律",
        "medical",
        "diagnosis",
        "prescription",
        "隐私",
        "compliance",
        "security incident",
    ]

    if any(keyword in lower for keyword in ("总结", "摘要", "summar", "overview")):
        question_type = "summarization"
    elif any(keyword in lower for keyword in ("对比", "区别", "compare", "vs")):
        question_type = "comparison"
    elif any(keyword in lower for keyword in ("分析", "原因", "why", "tradeoff", "影响")):
        question_type = "analysis"
    elif any(keyword in lower for keyword in ("什么", "怎么", "如何", "what", "how", "when", "where")):
        question_type = "faq"
    else:
        question_type = "document_qa"

    route = "internal_api" if any(keyword in lower for keyword in internal_keywords) else "rag"
    risk_level = "high" if any(keyword.lower() in lower for keyword in high_risk_keywords) else "low"
    return {
        "question_type": question_type,
        "route": route,
        "risk_level": risk_level,
        "needs_human_review": risk_level == "high",
        "reason": "heuristic fallback",
    }


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


def classify_question(state: RAGState) -> RAGState:
    """Classify the question type, route, and review risk."""
    trace = _append_trace(state, "Classifying question and risk level")
    llm = _get_llm()
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are an orchestration classifier for an agentic RAG system. "
                    "Return JSON only with keys: question_type, route, risk_level, needs_human_review, reason. "
                    "question_type must be one of faq, document_qa, summarization, comparison, analysis. "
                    "route must be one of rag, internal_api, direct. "
                    "risk_level must be one of low, medium, high."
                ),
            ),
            ("human", "Question: {question}"),
        ]
    )

    fallback = _heuristic_classification(state["question"])
    try:
        raw = (prompt | llm | StrOutputParser()).invoke({"question": state["question"]})
        payload = _extract_json_object(raw)
    except Exception:
        payload = fallback

    question_type = str(payload.get("question_type") or fallback["question_type"])
    route = str(payload.get("route") or fallback["route"])
    risk_level = str(payload.get("risk_level") or fallback["risk_level"])
    needs_human_review = bool(payload.get("needs_human_review", risk_level == "high"))
    trace.append(
        f"{len(trace) + 1}. Classified as type={question_type}, route={route}, risk={risk_level}"
    )
    logger.info(
        "[rag.classify] type=%s route=%s risk=%s human_review=%s",
        question_type,
        route,
        risk_level,
        needs_human_review,
    )
    return {
        **state,
        "question_type": question_type,
        "route": route,
        "risk_level": risk_level,
        "needs_human_review": needs_human_review,
        "trace": trace,
    }


def plan_question(state: RAGState) -> RAGState:
    """Create a compact execution plan and tool list."""
    trace = _append_trace(state, "Planning answer steps and candidate tools")
    llm = _get_llm()
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are a task planner for an agentic RAG workflow. "
                    "Return JSON only with keys: steps and tools. "
                    "steps must be a short list of 2-4 concise actions. "
                    "tools must be a short list chosen from knowledge_base_retrieval, internal_api_lookup, answer_validation, human_review."
                ),
            ),
            (
                "human",
                "Question: {question}\nType: {question_type}\nPreferred route: {route}",
            ),
        ]
    )

    fallback_steps = {
        "faq": ["Understand the request", "Retrieve grounding context", "Answer concisely", "Validate the answer"],
        "document_qa": ["Rewrite the query", "Retrieve relevant chunks", "Answer with cited evidence", "Validate the answer"],
        "summarization": ["Retrieve the most relevant chunks", "Condense the main points", "Check coverage and faithfulness"],
        "comparison": ["Retrieve evidence for each side", "Compare similarities and differences", "Validate the comparison"],
        "analysis": ["Retrieve supporting evidence", "Explain causes or tradeoffs", "Validate the reasoning"],
    }
    fallback_tools = [
        "internal_api_lookup" if state.get("route") == "internal_api" else "knowledge_base_retrieval",
        "answer_validation",
    ]

    try:
        raw = (prompt | llm | StrOutputParser()).invoke(
            {
                "question": state["question"],
                "question_type": state["question_type"],
                "route": state["route"],
            }
        )
        payload = _extract_json_object(raw)
        plan_steps = [str(step) for step in payload.get("steps", []) if str(step).strip()]
        planned_tools = [str(tool) for tool in payload.get("tools", []) if str(tool).strip()]
    except Exception:
        plan_steps = fallback_steps.get(state["question_type"], fallback_steps["document_qa"])
        planned_tools = fallback_tools

    if not plan_steps:
        plan_steps = fallback_steps.get(state["question_type"], fallback_steps["document_qa"])
    if not planned_tools:
        planned_tools = fallback_tools

    trace.append(f"{len(trace) + 1}. Planned {len(plan_steps)} execution steps")
    logger.info("[rag.plan] steps=%s tools=%s", plan_steps, planned_tools)
    return {**state, "plan_steps": plan_steps, "planned_tools": planned_tools, "trace": trace}


def call_internal_api(state: RAGState) -> RAGState:
    """Use internal service metadata as a lightweight tool."""
    trace = _append_trace(state, "Calling internal metadata tool")
    collections = list_collections = None
    from app.services.indexing import list_collections as _list_collections

    collections = _list_collections()
    collection_summary = ", ".join(f"{item['name']}({item['count']})" for item in collections[:10])
    tool_context = [
        f"Active LLM model: {settings.llm_model}",
        f"Active embedding model: {settings.embedding_model}",
        f"Collections: {collection_summary or 'none'}",
    ]
    tool_calls = _append_tool_call(
        state,
        name="internal_api_lookup",
        status="success",
        input_summary=state["question"][:120],
        output_summary=tool_context[-1][:200],
    )
    trace.append(f"{len(trace) + 1}. Internal metadata tool returned {len(tool_context)} facts")
    logger.info("[rag.tool.internal_api] facts=%s", len(tool_context))
    return {**state, "tool_context": tool_context, "tool_calls": tool_calls, "trace": trace}


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

    tool_calls = _append_tool_call(
        state,
        name="knowledge_base_retrieval",
        status="success",
        input_summary=(state.get("current_query") or state["question"])[:120],
        output_summary=f"Retrieved {len(docs)} chunks in hop {current_hop}",
    )
    logger.info("[rag.retrieve] complete docs=%s scores=%s", len(docs), scores)
    trace.append(
        f"{len(trace) + 1}. Retrieval hop {current_hop} complete: {len(docs)} candidate chunks found"
    )
    if scores:
        trace.append(f"{len(trace) + 1}. Retrieval hop {current_hop} scores: {', '.join(scores)}")
    return {
        **state,
        "candidate_documents": docs,
        "retrieval_hop": current_hop,
        "tool_calls": tool_calls,
        "trace": trace,
    }


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
    context_parts = [doc.page_content for doc in state["documents"]]
    if state.get("tool_context"):
        context_parts.extend(state["tool_context"])
    context = "\n\n---\n\n".join(context_parts)

    question_type = state.get("question_type", "document_qa")
    answer_style = {
        "faq": "Answer directly and concisely.",
        "document_qa": "Answer directly and ground every claim in the provided context.",
        "summarization": "Produce a concise summary of the main points.",
        "comparison": "Compare the relevant options with similarities and differences.",
        "analysis": "Provide a structured analysis with reasoning and tradeoffs.",
    }.get(question_type, "Answer directly and ground every claim in the provided context.")

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are a helpful assistant. "
                    "Answer the user's question based solely on the provided context. "
                    "If the context does not contain enough information, say so honestly. "
                    "Follow this answer style instruction: {answer_style}"
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
    answer = chain.invoke(
        {
            "context": context or "(no context available)",
            "question": state["question"],
            "answer_style": answer_style,
        }
    )
    logger.info("[rag.generate] complete answer_chars=%s", len(answer))
    trace.append(f"{len(trace) + 1}. Answer generation complete")
    return {**state, "answer": answer, "trace": trace}


def validate_answer(state: RAGState) -> RAGState:
    """Validate the answer against retrieved evidence and tool outputs."""
    trace = _append_trace(state, "Validating answer confidence and grounding")
    llm = _get_llm()
    support_context = [doc.page_content[: settings.source_preview_chars] for doc in state.get("documents", [])]
    support_context.extend(state.get("tool_context", []))
    support_text = "\n\n---\n\n".join(support_context) or "(no support context)"

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You validate answers in an agentic RAG system. "
                    "Return JSON only with keys: passed, confidence, citations_verified, issues. "
                    "passed must be true only when the answer is well supported by the provided evidence. "
                    "confidence must be a number between 0 and 1. "
                    "citations_verified should be true when the answer is grounded in the evidence supplied."
                ),
            ),
            (
                "human",
                "Question: {question}\n\nAnswer: {answer}\n\nEvidence:\n{evidence}",
            ),
        ]
    )

    fallback = {
        "passed": bool(state.get("answer") and support_context),
        "confidence": 0.75 if support_context else 0.25,
        "citations_verified": bool(support_context),
        "issues": [] if support_context else ["No grounding evidence available for validation."],
    }
    try:
        raw = (prompt | llm | StrOutputParser()).invoke(
            {
                "question": state["question"],
                "answer": state.get("answer", ""),
                "evidence": support_text,
            }
        )
        payload = _extract_json_object(raw)
    except Exception:
        payload = fallback

    confidence = payload.get("confidence", fallback["confidence"])
    try:
        confidence_value = round(float(confidence), 4)
    except (TypeError, ValueError):
        confidence_value = float(fallback["confidence"])

    passed = bool(payload.get("passed", fallback["passed"])) and (
        confidence_value >= settings.answer_validation_min_confidence
    )
    citations_verified = bool(payload.get("citations_verified", fallback["citations_verified"]))
    raw_issues = payload.get("issues", fallback["issues"]) or fallback["issues"]
    if isinstance(raw_issues, str):
        normalized_issues = [raw_issues]
    else:
        normalized_issues = list(raw_issues)
    issues = [str(item) for item in normalized_issues if str(item).strip()]
    tool_calls = _append_tool_call(
        state,
        name="answer_validation",
        status="success" if passed else "warning",
        input_summary=state["question"][:120],
        output_summary=f"passed={passed}, confidence={confidence_value}",
    )
    trace.append(
        f"{len(trace) + 1}. Validation complete: passed={passed}, confidence={confidence_value:.2f}"
    )
    logger.info(
        "[rag.validate] passed=%s confidence=%s citations_verified=%s issues=%s",
        passed,
        confidence_value,
        citations_verified,
        issues,
    )
    return {
        **state,
        "confidence_score": confidence_value,
        "validation_passed": passed,
        "validation_issues": issues,
        "citations_verified": citations_verified,
        "tool_calls": tool_calls,
        "trace": trace,
    }


def reflect_and_retry(state: RAGState) -> RAGState:
    """Retry when confidence is low by looping back into retrieval."""
    next_retry = state.get("retry_count", 0) + 1
    trace = _append_trace(
        state,
        f"Low-confidence answer detected; retrying with another retrieval cycle (attempt {next_retry})",
    )
    logger.info("[rag.reflect] retry=%s issues=%s", next_retry, state.get("validation_issues", []))
    return {**state, "retry_count": next_retry, "trace": trace}


def human_review(state: RAGState) -> RAGState:
    """Flag the result for human review when risk or confidence requires it."""
    reason = state.get("human_review_reason") or "High-risk request or low-confidence answer requires human confirmation."
    trace = _append_trace(state, f"Marked for human review: {reason}")
    answer = state.get("answer", "")
    if answer:
        answer = f"Draft answer generated, but human review is required before relying on it.\n\n{answer}"
    else:
        answer = "Human review is required before this request can be answered safely."
    logger.info("[rag.human_review] reason=%s", reason)
    return {
        **state,
        "needs_human_review": True,
        "human_review_reason": reason,
        "answer": answer,
        "trace": trace,
    }


def finalize(state: RAGState) -> RAGState:
    trace = _append_trace(state, "Response finalized")
    return {**state, "trace": trace}


def no_answer(state: RAGState) -> RAGState:
    """Return a polite fallback when no relevant documents were found."""
    trace = _append_trace(state, "No relevant chunks remained after grading")
    logger.info("[rag.no_answer] no relevant documents found")
    needs_human_review = bool(state.get("needs_human_review") or state.get("risk_level") == "high")
    human_review_reason = state.get("human_review_reason") or (
        "High-risk request has no grounded evidence and requires human confirmation."
        if needs_human_review
        else ""
    )
    return {
        **state,
        "answer": (
            "I could not find any relevant information in the knowledge base "
            "to answer your question."
        ),
        "confidence_score": state.get("confidence_score") if state.get("confidence_score") is not None else 0.0,
        "validation_passed": False,
        "validation_issues": state.get("validation_issues") or [
            "No relevant grounded evidence was found for this request."
        ],
        "citations_verified": False,
        "needs_human_review": needs_human_review,
        "human_review_reason": human_review_reason,
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


def _route_after_planning(state: RAGState) -> str:
    if state.get("route") == "internal_api":
        return "call_internal_api"
    if state.get("route") == "direct":
        return "generate"
    return "rewrite_query"


def _route_after_validation(state: RAGState) -> str:
    if state.get("risk_level") == "high":
        return "human_review"
    if state.get("validation_passed"):
        return "finalize"
    if state.get("route") == "rag" and state.get("retry_count", 0) < settings.max_validation_retries:
        return "reflect_and_retry"
    if state.get("needs_human_review") or state.get("risk_level") in {"medium", "high"}:
        return "human_review"
    return "finalize"


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------


def build_rag_graph() -> StateGraph:
    """Compile and return the RAG StateGraph."""
    logger.info("[rag.graph] building graph")
    graph = StateGraph(RAGState)

    graph.add_node("classify_question", classify_question)
    graph.add_node("plan_question", plan_question)
    graph.add_node("call_internal_api", call_internal_api)
    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("retrieve", retrieve)
    graph.add_node("grade_docs", grade_docs)
    graph.add_node("generate", generate)
    graph.add_node("validate_answer", validate_answer)
    graph.add_node("reflect_and_retry", reflect_and_retry)
    graph.add_node("human_review", human_review)
    graph.add_node("finalize", finalize)
    graph.add_node("no_answer", no_answer)

    graph.add_edge(START, "classify_question")
    graph.add_edge("classify_question", "plan_question")
    graph.add_conditional_edges(
        "plan_question",
        _route_after_planning,
        {
            "call_internal_api": "call_internal_api",
            "generate": "generate",
            "rewrite_query": "rewrite_query",
        },
    )
    graph.add_edge("call_internal_api", "generate")
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
    graph.add_edge("generate", "validate_answer")
    graph.add_conditional_edges(
        "validate_answer",
        _route_after_validation,
        {
            "reflect_and_retry": "reflect_and_retry",
            "human_review": "human_review",
            "finalize": "finalize",
        },
    )
    graph.add_edge("reflect_and_retry", "rewrite_query")
    graph.add_edge("finalize", END)
    graph.add_edge("human_review", END)
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
