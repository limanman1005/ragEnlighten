"""Plan-first Agent execution service.

This module implements a two-phase question-answering flow:
1. create an explicit plan;
2. execute the completed plan in order.
"""

from __future__ import annotations

import inspect
import json
import logging
import re
from dataclasses import dataclass
from typing import Any, AsyncIterator, Callable

from app.core.config import settings
logger = logging.getLogger("uvicorn.error")

_ALLOWED_PLAN_TOOLS = {"knowledge_base_search", "collection_overview", "answer_synthesis"}


@dataclass(frozen=True)
class PlanStep:
    step_id: str
    purpose: str
    tool: str
    query: str = ""


try:
    from langchain_core.documents import Document
except ModuleNotFoundError:
    @dataclass
    class Document:  # type: ignore[no-redef]
        page_content: str
        metadata: dict[str, Any]


@dataclass
class PlanExecutionResult:
    documents: list[Document]
    tool_calls: list[dict[str, str]]
    tool_context: list[str]
    trace: list[str]
    debug_events: list[dict[str, Any]]


def _append_debug_event(
    events: list[dict[str, Any]],
    phase: str,
    message: str,
    **details: Any,
) -> list[dict[str, Any]]:
    entry = {"phase": phase, "message": message}
    if details:
        entry["details"] = details
    events.append(entry)
    return events


def _get_llm():
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=settings.llm_model,
        temperature=0,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
    )


def _extract_json_object(text: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in model response")
    return json.loads(match.group(0))


def _build_fallback_plan(question: str) -> list[PlanStep]:
    return [
        PlanStep(
            step_id="1",
            purpose="Search the knowledge base for evidence relevant to the question",
            tool="knowledge_base_search",
            query=question,
        ),
        PlanStep(
            step_id="2",
            purpose="Synthesize a grounded answer from the completed tool outputs",
            tool="answer_synthesis",
            query="Answer using only the retrieved evidence and tool outputs.",
        ),
    ]


def _normalize_plan_tool(tool: str) -> str:
    normalized = tool.strip().lower()
    if normalized in {"search", "retrieval", "retrieve", "knowledge_base_retrieval"}:
        return "knowledge_base_search"
    if normalized in {"metadata", "internal_api", "system_info"}:
        return "collection_overview"
    if normalized in {"synthesize", "generate", "answer"}:
        return "answer_synthesis"
    return normalized


def _parse_plan_steps(raw: str, question: str) -> list[PlanStep]:
    try:
        payload = _extract_json_object(raw)
        raw_steps = payload.get("steps")
        if not isinstance(raw_steps, list):
            raise ValueError("Plan JSON must contain a steps list")

        steps: list[PlanStep] = []
        for index, item in enumerate(raw_steps, start=1):
            if not isinstance(item, dict):
                continue
            tool = _normalize_plan_tool(str(item.get("tool") or "knowledge_base_search"))
            if tool not in _ALLOWED_PLAN_TOOLS:
                continue
            purpose = str(item.get("purpose") or item.get("description") or "").strip()
            query = str(item.get("query") or item.get("instruction") or "").strip()
            steps.append(
                PlanStep(
                    step_id=str(item.get("id") or index),
                    purpose=purpose or f"Execute step {index}",
                    tool=tool,
                    query=query or question,
                )
            )
        if not steps:
            raise ValueError("Plan contains no executable steps")
        if steps[-1].tool != "answer_synthesis":
            steps.append(
                PlanStep(
                    step_id=str(len(steps) + 1),
                    purpose="Synthesize a grounded final answer",
                    tool="answer_synthesis",
                    query="Answer using only the completed plan outputs.",
                )
            )
        return steps
    except Exception:
        return _build_fallback_plan(question)


def _plan_to_strings(steps: list[PlanStep]) -> list[str]:
    return [f"{step.step_id}. {step.purpose} ({step.tool})" for step in steps]


def _build_planning_prompt():
    from langchain_core.prompts import ChatPromptTemplate

    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are a planning module for a plan-and-execute RAG agent. "
                    "Create the full execution plan before any tools are used. "
                    "Return JSON only with key steps. Each step must have id, purpose, tool, and query. "
                    "Allowed tools: knowledge_base_search, collection_overview, answer_synthesis. "
                    "Use knowledge_base_search for document or knowledge-base questions. "
                    "Use collection_overview only for questions about model configuration, collections, or service metadata. "
                    "The last step should be answer_synthesis."
                ),
            ),
            (
                "human",
                "Question: {question}\n\nChat history summary:\n{history}\n\nPlan JSON:",
            ),
        ]
    )


def _format_history(history: list[dict[str, Any]] | None) -> str:
    if not history:
        return "(none)"
    parts = []
    for item in history[-8:]:
        role = item.get("role", "user")
        content = str(item.get("content", "")).strip()
        if content:
            parts.append(f"{role}: {content[:500]}")
    return "\n".join(parts) or "(none)"


def _create_plan(question: str, history: list[dict[str, Any]] | None = None) -> list[PlanStep]:
    from langchain_core.output_parsers import StrOutputParser

    prompt = _build_planning_prompt()
    try:
        raw = (prompt | _get_llm() | StrOutputParser()).invoke(
            {"question": question, "history": _format_history(history)}
        )
        return _parse_plan_steps(raw, question)
    except Exception:
        logger.exception("[plan_execute.plan] planning failed; using fallback")
        return _build_fallback_plan(question)


def _merge_documents(existing: list[Document], incoming: list[Document]) -> list[Document]:
    merged = list(existing)
    seen = {
        (
            doc.metadata.get("source"),
            doc.metadata.get("page"),
            doc.metadata.get("start_index"),
            doc.page_content,
        )
        for doc in merged
    }
    for doc in incoming:
        key = (
            doc.metadata.get("source"),
            doc.metadata.get("page"),
            doc.metadata.get("start_index"),
            doc.page_content,
        )
        if key not in seen:
            seen.add(key)
            merged.append(doc)
    return merged


def _format_parent_child_documents(child_docs: list[Document], parent_docs: list[Document]) -> str:
    if not child_docs:
        return "No relevant documents were found in the knowledge base."

    parent_by_id = {
        str(doc.metadata.get("parent_chunk_id") or ""): doc
        for doc in parent_docs
        if doc.metadata.get("parent_chunk_id")
    }
    parts: list[str] = []
    for index, child_doc in enumerate(child_docs, start=1):
        parent_id = str(child_doc.metadata.get("parent_chunk_id") or "")
        parent_doc = parent_by_id.get(parent_id)
        source = child_doc.metadata.get("source", "unknown")
        section_path = child_doc.metadata.get("section_path") or "root"
        score = child_doc.metadata.get("retrieval_score", "n/a")
        child_preview = (
            child_doc.metadata.get("content_preview") or child_doc.page_content
        )[: settings.source_preview_chars]
        if parent_doc is None:
            parts.append(f"[{index}] source={source}; section={section_path}; score={score}\n{child_preview}")
            continue
        parent_preview = (
            parent_doc.metadata.get("content_preview") or parent_doc.page_content
        )[: settings.rewrite_context_chars]
        parts.append(
            f"[{index}] source={source}; section={section_path}; score={score}\n"
            f"Parent context:\n{parent_preview}\nMatched child evidence:\n{child_preview}"
        )
    return "\n\n".join(parts)


def _knowledge_base_search(query: str, collection_name: str | None = None) -> tuple[str, list[Document]]:
    from app.services.indexing import get_parent_chunks_by_ids, get_vectorstore

    vectorstore = get_vectorstore(collection_name)
    scored_docs = vectorstore.similarity_search_with_relevance_scores(
        query=query,
        k=settings.retriever_top_k,
        filter={"chunk_level": "child"},
    )
    child_docs: list[Document] = []
    parent_chunk_ids: list[str] = []
    for doc, score in scored_docs:
        metadata = dict(doc.metadata)
        metadata["retrieval_score"] = round(float(score), 4)
        metadata["retrieval_hop"] = 1
        metadata["retrieval_query"] = query
        child_doc = Document(page_content=doc.page_content, metadata=metadata)
        child_docs.append(child_doc)
        parent_chunk_id = str(metadata.get("parent_chunk_id") or "").strip()
        if parent_chunk_id:
            parent_chunk_ids.append(parent_chunk_id)

    parent_docs = get_parent_chunks_by_ids(parent_chunk_ids, collection_name)
    parent_by_id = {
        str(doc.metadata.get("parent_chunk_id") or ""): doc
        for doc in parent_docs
        if doc.metadata.get("parent_chunk_id")
    }
    enriched_docs: list[Document] = []
    for child_doc in child_docs:
        metadata = dict(child_doc.metadata)
        parent_doc = parent_by_id.get(str(metadata.get("parent_chunk_id") or ""))
        if parent_doc is not None:
            metadata["parent_title"] = parent_doc.metadata.get("title") or parent_doc.metadata.get("section_path")
            metadata["parent_section_path"] = parent_doc.metadata.get("section_path") or parent_doc.metadata.get("parent_section_path")
            metadata["parent_content_preview"] = (
                parent_doc.metadata.get("content_preview") or parent_doc.page_content
            )[: settings.rewrite_context_chars]
        enriched_docs.append(Document(page_content=child_doc.page_content, metadata=metadata))

    return _format_parent_child_documents(enriched_docs, parent_docs), enriched_docs


def _collection_overview() -> str:
    from app.services.indexing import list_collections

    collections = list_collections()
    summary = ", ".join(f"{item['name']}({item['count']})" for item in collections[:10]) or "none"
    return (
        f"Active LLM model: {settings.llm_model}\n"
        f"Active embedding model: {settings.embedding_model}\n"
        f"Default collection: {settings.chroma_collection_name}\n"
        f"Collections: {summary}"
    )


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


async def _execute_plan_steps(
    steps: list[PlanStep],
    *,
    search: Callable[[str], Any],
    overview: Callable[[], Any],
) -> PlanExecutionResult:
    documents: list[Document] = []
    tool_calls: list[dict[str, str]] = []
    tool_context: list[str] = []
    trace: list[str] = []
    debug_events: list[dict[str, Any]] = []

    for step in steps:
        if step.tool == "answer_synthesis":
            trace.append(f"{len(trace) + 1}. Plan step {step.step_id}: ready for answer synthesis")
            continue

        trace.append(f"{len(trace) + 1}. Executing plan step {step.step_id}: {step.purpose}")
        _append_debug_event(
            debug_events,
            "execution",
            f"Executing plan step {step.step_id}",
            tool=step.tool,
            query=step.query,
        )
        if step.tool == "knowledge_base_search":
            output, docs = await _maybe_await(search(step.query))
            documents = _merge_documents(documents, docs)
        elif step.tool == "collection_overview":
            output = await _maybe_await(overview())
        else:
            continue

        tool_context.append(str(output))
        tool_calls.append(
            {
                "name": step.tool,
                "status": "success",
                "input_summary": step.query[:120],
                "output_summary": str(output)[:1600],
            }
        )
        trace.append(f"{len(trace) + 1}. Tool {step.tool} returned a result")

    return PlanExecutionResult(
        documents=documents,
        tool_calls=tool_calls,
        tool_context=tool_context,
        trace=trace,
        debug_events=debug_events,
    )


def _synthesize_answer(question: str, plan: list[PlanStep], execution: PlanExecutionResult) -> str:
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    context = "\n\n---\n\n".join(execution.tool_context) or "(no tool output available)"
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are the synthesis phase of a plan-and-execute RAG agent. "
                    "Do not call tools or invent new steps. Answer only from the completed plan outputs. "
                    "If the outputs are insufficient, say so directly."
                ),
            ),
            (
                "human",
                "Question: {question}\n\nCompleted plan:\n{plan}\n\nPlan outputs:\n{context}\n\nAnswer:",
            ),
        ]
    )
    return (prompt | _get_llm() | StrOutputParser()).invoke(
        {
            "question": question,
            "plan": "\n".join(_plan_to_strings(plan)),
            "context": context,
        }
    )


def _has_supporting_context(execution: PlanExecutionResult) -> bool:
    return bool(execution.documents or execution.tool_context)


def _build_validation(execution: PlanExecutionResult) -> dict[str, Any]:
    has_grounding = _has_supporting_context(execution)
    return {
        "passed": has_grounding,
        "confidence": (
            min(0.95, 0.35 + (0.15 * len(execution.documents)))
            if execution.documents
            else (0.7 if execution.tool_context else 0.2)
        ),
        "citations_verified": has_grounding,
        "issues": [] if has_grounding else ["No supporting context was returned by the plan execution tools."],
    }


def _apply_grounding_guard(answer: str, execution: PlanExecutionResult) -> tuple[str, bool, str | None]:
    if _has_supporting_context(execution):
        return answer, False, None
    return (
        "I could not find grounded evidence in the knowledge base for this question, so I am not returning a factual answer. "
        "Please refine the query or inspect the plan execution trace.",
        True,
        "No supporting context was returned by the plan execution tools.",
    )


def _build_result(
    question: str,
    plan: list[PlanStep],
    execution: PlanExecutionResult,
    answer: str,
    needs_human_review: bool,
    human_review_reason: str | None,
) -> dict[str, Any]:
    validation = _build_validation(execution)
    return {
        "question": question,
        "answer": answer,
        "question_type": "document_qa",
        "route": "plan_execute_agent",
        "plan": _plan_to_strings(plan),
        "tool_calls": execution.tool_calls,
        "documents": execution.documents,
        "trace": execution.trace,
        "debug_events": execution.debug_events,
        "confidence_score": round(float(validation["confidence"]), 4),
        "needs_human_review": needs_human_review,
        "human_review_reason": human_review_reason,
        "validation": validation,
    }


async def run_plan_execute_agent_query(
    question: str,
    collection_name: str | None = None,
    history: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    trace = ["1. Query accepted by Plan-Execute Agent API", "2. Planning phase started"]
    debug_events: list[dict[str, Any]] = []
    _append_debug_event(debug_events, "planning", "Planning phase started")
    plan = _create_plan(question, history)
    trace.append(f"3. Planning complete: {len(plan)} steps")
    _append_debug_event(debug_events, "plan", "Planning phase complete", plan=_plan_to_strings(plan))

    execution = await _execute_plan_steps(
        plan,
        search=lambda query: _knowledge_base_search(query, collection_name),
        overview=_collection_overview,
    )
    execution.trace = [*trace, *execution.trace]
    execution.debug_events = [*debug_events, *execution.debug_events]
    raw_answer = _synthesize_answer(question, plan, execution)
    answer, needs_human_review, human_review_reason = _apply_grounding_guard(
        raw_answer,
        execution,
    )
    execution.trace.append(f"{len(execution.trace) + 1}. Final answer synthesized from completed plan")
    return _build_result(
        question,
        plan,
        execution,
        answer,
        needs_human_review,
        human_review_reason,
    )


async def stream_plan_execute_agent_query(
    question: str,
    collection_name: str | None = None,
    history: list[dict[str, str]] | None = None,
) -> AsyncIterator[dict[str, Any]]:
    yield {"type": "debug", "data": {"phase": "planning", "message": "Planning phase started"}}
    yield {"type": "trace", "data": "1. Query accepted by Plan-Execute Agent stream API"}
    yield {"type": "trace", "data": "2. Planning phase started"}
    plan = _create_plan(question, history)
    plan_strings = _plan_to_strings(plan)
    yield {"type": "plan", "data": plan_strings}
    yield {"type": "debug", "data": {"phase": "plan", "message": "Planning phase complete", "details": {"plan": plan_strings}}}

    execution = await _execute_plan_steps(
        plan,
        search=lambda query: _knowledge_base_search(query, collection_name),
        overview=_collection_overview,
    )
    for item in execution.trace:
        yield {"type": "trace", "data": item}
    for event in execution.debug_events:
        yield {"type": "debug", "data": event}

    raw_answer = _synthesize_answer(question, plan, execution)
    answer, needs_human_review, human_review_reason = _apply_grounding_guard(
        raw_answer,
        execution,
    )
    final_trace = [*["1. Query accepted by Plan-Execute Agent stream API", "2. Planning phase started"], *execution.trace]
    final_trace.append(f"{len(final_trace) + 1}. Final answer synthesized from completed plan")
    execution.trace = final_trace
    result = _build_result(
        question,
        plan,
        execution,
        answer,
        needs_human_review,
        human_review_reason,
    )
    yield {"type": "final", "data": result}
