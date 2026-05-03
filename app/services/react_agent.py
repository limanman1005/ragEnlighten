"""React Agent-based chat execution service."""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from app.core.config import settings
from app.services.indexing import get_parent_chunks_by_ids, get_vectorstore, list_collections


logger = logging.getLogger("uvicorn.error")


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


def _log_react_stage(stage: str, round_number: int | None, message: str) -> None:
    if round_number is None:
        logger.info("[react_agent.stage] stage=%s %s", stage, message)
        return
    logger.info("[react_agent.stage] round=%s stage=%s %s", round_number, stage, message)


class DeepSeekReasoningChatOpenAI(ChatOpenAI):
    """Preserve DeepSeek reasoning_content on assistant messages across agent turns."""

    def _get_request_payload(
        self,
        input_: Any,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        messages = self._convert_input(input_).to_messages()
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        if self._use_responses_api(payload):
            logger.info(
                "[react_agent.llm.payload] responses_api=true messages=%s model=%s",
                len(messages),
                self.model_name,
            )
            return payload

        injected_reasoning = 0
        for original_message, payload_message in zip(messages, payload.get("messages", [])):
            if not isinstance(original_message, AIMessage):
                continue
            reasoning_content = original_message.additional_kwargs.get("reasoning_content")
            if reasoning_content:
                payload_message["reasoning_content"] = reasoning_content
                injected_reasoning += 1
        logger.info(
            "[react_agent.llm.payload] messages=%s assistant_reasoning_injected=%s model=%s",
            len(payload.get("messages", [])),
            injected_reasoning,
            self.model_name,
        )
        return payload

    def _create_chat_result(
        self,
        response: dict | Any,
        generation_info: dict | None = None,
    ) -> ChatResult:
        result = super()._create_chat_result(response, generation_info)
        response_dict = response if isinstance(response, dict) else response.model_dump()
        extracted_reasoning = 0
        for generation, choice in zip(result.generations, response_dict.get("choices", [])):
            message = choice.get("message", {}) or {}
            reasoning_content = message.get("reasoning_content")
            if reasoning_content and isinstance(generation.message, AIMessage):
                generation.message.additional_kwargs["reasoning_content"] = reasoning_content
                extracted_reasoning += 1
        logger.info(
            "[react_agent.llm.result] generations=%s reasoning_extracted=%s model=%s",
            len(result.generations),
            extracted_reasoning,
            self.model_name,
        )
        return result

    def _convert_chunk_to_generation_chunk(
        self,
        chunk: dict,
        default_chunk_class: type,
        base_generation_info: dict | None,
    ) -> ChatGenerationChunk | None:
        generation_chunk = super()._convert_chunk_to_generation_chunk(
            chunk,
            default_chunk_class,
            base_generation_info,
        )
        if generation_chunk is None:
            return None

        choices = chunk.get("choices", []) or chunk.get("chunk", {}).get("choices", [])
        if not choices:
            return generation_chunk

        delta = choices[0].get("delta") or {}
        reasoning_content = delta.get("reasoning_content")
        if reasoning_content and isinstance(generation_chunk.message, AIMessageChunk):
            generation_chunk.message.additional_kwargs["reasoning_content"] = reasoning_content
            logger.info(
                "[react_agent.llm.stream_chunk] reasoning_chunk=true model=%s",
                self.model_name,
            )
        return generation_chunk


def _get_llm() -> ChatOpenAI:
    llm_cls = DeepSeekReasoningChatOpenAI if "api.deepseek.com" in settings.llm_base_url else ChatOpenAI
    llm = llm_cls(
        model=settings.llm_model,
        temperature=0,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
    )
    logger.info(
        "[react_agent.llm.init] class=%s model=%s base_url=%s",
        llm.__class__.__name__,
        settings.llm_model,
        settings.llm_base_url,
    )
    return llm


def _stringify_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content") or ""
                if text:
                    parts.append(str(text))
        return "\n".join(part for part in parts if part).strip()
    if content is None:
        return ""
    return str(content)


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


def _format_documents(docs: list[Document]) -> str:
    if not docs:
        return "No relevant documents were found in the knowledge base."

    parts: list[str] = []
    for index, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        section_path = doc.metadata.get("section_path") or "root"
        retrieval_score = doc.metadata.get("retrieval_score")
        score_text = f"{float(retrieval_score):.4f}" if retrieval_score is not None else "n/a"
        snippet = (doc.metadata.get("content_preview") or doc.page_content)[: settings.source_preview_chars]
        parts.append(
            f"[{index}] source={source}; section={section_path}; score={score_text}\n{snippet}"
        )
    return "\n\n".join(parts)


def _format_parent_child_documents(
    child_docs: list[Document],
    parent_docs: list[Document],
) -> str:
    if not child_docs:
        return "No relevant documents were found in the knowledge base."

    parent_by_id = {
        str(doc.metadata.get("parent_chunk_id") or ""): doc
        for doc in parent_docs
        if doc.metadata.get("parent_chunk_id")
    }
    grouped_children: dict[str, list[Document]] = {}
    for child_doc in child_docs:
        parent_id = str(child_doc.metadata.get("parent_chunk_id") or "")
        grouped_children.setdefault(parent_id, []).append(child_doc)

    parts: list[str] = []
    for group_index, child_doc in enumerate(child_docs, start=1):
        parent_id = str(child_doc.metadata.get("parent_chunk_id") or "")
        if group_index > 1 and parent_id in grouped_children:
            if grouped_children[parent_id][0] is not child_doc:
                continue

        source = child_doc.metadata.get("source", "unknown")
        section_path = child_doc.metadata.get("section_path") or "root"
        retrieval_score = child_doc.metadata.get("retrieval_score")
        score_text = f"{float(retrieval_score):.4f}" if retrieval_score is not None else "n/a"
        matched_children = grouped_children.get(parent_id, [child_doc])
        parent_doc = parent_by_id.get(parent_id)

        header = f"[{group_index}] source={source}; section={section_path}; score={score_text}"
        if parent_doc is not None:
            parent_title = parent_doc.metadata.get("title") or parent_doc.metadata.get("section_path") or "root"
            parent_preview = (
                parent_doc.metadata.get("content_preview") or parent_doc.page_content
            )[: settings.rewrite_context_chars]
            child_parts = []
            for child_index, matched_child in enumerate(matched_children, start=1):
                child_preview = (
                    matched_child.metadata.get("content_preview") or matched_child.page_content
                )[: settings.source_preview_chars]
                child_parts.append(f"  Child {child_index}: {child_preview}")
            parts.append(
                f"{header}\nParent context [{parent_title}]:\n{parent_preview}\nMatched child evidence:\n"
                + "\n".join(child_parts)
            )
        else:
            child_preview = (
                child_doc.metadata.get("content_preview") or child_doc.page_content
            )[: settings.source_preview_chars]
            parts.append(f"{header}\n{child_preview}")
    return "\n\n".join(parts)


def _build_retrieval_audit_text(
    child_docs: list[Document],
    parent_docs: list[Document],
) -> str:
    if not child_docs:
        return "Retrieval audit: no child chunks matched the query."

    child_lines: list[str] = []
    for index, child_doc in enumerate(child_docs, start=1):
        child_lines.append(
            "- "
            f"child[{index}] source={child_doc.metadata.get('source', 'unknown')}; "
            f"title={child_doc.metadata.get('title') or 'n/a'}; "
            f"section={child_doc.metadata.get('section_path') or 'root'}; "
            f"parent_chunk_id={child_doc.metadata.get('parent_chunk_id') or 'n/a'}; "
            f"score={child_doc.metadata.get('retrieval_score', 'n/a')}"
        )

    parent_lines: list[str] = []
    for index, parent_doc in enumerate(parent_docs, start=1):
        preview = (parent_doc.metadata.get('content_preview') or parent_doc.page_content)[:200]
        parent_lines.append(
            "- "
            f"parent[{index}] title={parent_doc.metadata.get('title') or 'n/a'}; "
            f"section={parent_doc.metadata.get('section_path') or parent_doc.metadata.get('parent_section_path') or 'root'}; "
            f"parent_chunk_id={parent_doc.metadata.get('parent_chunk_id') or 'n/a'}; "
            f"preview={preview}"
        )

    parts = ["Retrieval audit", "Matched child chunks:", *child_lines]
    if parent_lines:
        parts.extend(["Recovered parent contexts:", *parent_lines])
    else:
        parts.append("Recovered parent contexts: none")
    return "\n".join(parts)


def _build_grounding_context_details(documents: list[Document]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for document in documents:
        metadata = document.metadata
        parent_chunk_id = str(metadata.get("parent_chunk_id") or f"no-parent-{len(grouped) + 1}")
        entry = grouped.setdefault(
            parent_chunk_id,
            {
                "source": metadata.get("source", "unknown"),
                "parent_chunk_id": metadata.get("parent_chunk_id"),
                "parent_title": metadata.get("parent_title"),
                "parent_section_path": metadata.get("parent_section_path"),
                "parent_content_preview": metadata.get("parent_content_preview"),
                "children": [],
            },
        )
        entry["children"].append(
            {
                "title": metadata.get("title"),
                "section_path": metadata.get("section_path"),
                "retrieval_score": metadata.get("retrieval_score"),
                "content_preview": (metadata.get("content_preview") or document.page_content)[: settings.source_preview_chars],
            }
        )
    return list(grouped.values())


def _build_grounding_context_text(documents: list[Document]) -> str:
    details = _build_grounding_context_details(documents)
    if not details:
        return "Final grounding context: no retrieved chunks were available for answer synthesis."

    parts = ["Final grounding context used for answer synthesis"]
    for index, entry in enumerate(details, start=1):
        parts.append(
            f"Group {index}: source={entry['source']}; parent_chunk_id={entry.get('parent_chunk_id') or 'n/a'}; "
            f"parent_title={entry.get('parent_title') or 'n/a'}; parent_section={entry.get('parent_section_path') or 'n/a'}"
        )
        if entry.get("parent_content_preview"):
            parts.append(f"  Parent preview: {entry['parent_content_preview']}")
        for child_index, child in enumerate(entry.get("children", []), start=1):
            parts.append(
                f"  Child {child_index}: title={child.get('title') or 'n/a'}; "
                f"section={child.get('section_path') or 'root'}; "
                f"score={child.get('retrieval_score', 'n/a')}; "
                f"preview={child.get('content_preview') or ''}"
            )
    return "\n".join(parts)


def _build_grounding_status(documents: list[Document], tool_calls: list[dict[str, str]]) -> dict[str, Any]:
    knowledge_search_calls = [call for call in tool_calls if call.get("name") == "knowledge_base_search"]
    metadata_calls = [call for call in tool_calls if call.get("name") == "collection_overview"]

    if documents:
        return {
            "status": "grounded",
            "message": (
                f"Grounded answer with {len(documents)} retrieved child chunks after "
                f"{len(knowledge_search_calls)} knowledge-base search call(s)."
            ),
            "knowledge_search_calls": len(knowledge_search_calls),
            "metadata_calls": len(metadata_calls),
            "documents": len(documents),
        }

    if not knowledge_search_calls:
        return {
            "status": "no_tool_call",
            "message": (
                "No retrieved chunks were available because the agent finished without calling "
                "knowledge_base_search. Any answer produced in this state is not grounded in the vector store."
            ),
            "knowledge_search_calls": 0,
            "metadata_calls": len(metadata_calls),
            "documents": 0,
        }

    return {
        "status": "empty_retrieval",
        "message": (
            f"knowledge_base_search was called {len(knowledge_search_calls)} time(s), but no child chunks were retained "
            "for final answer synthesis."
        ),
        "knowledge_search_calls": len(knowledge_search_calls),
        "metadata_calls": len(metadata_calls),
        "documents": 0,
    }


def _apply_grounding_guard(answer: str, grounding_status: dict[str, Any]) -> tuple[str, bool, str | None]:
    if grounding_status.get("status") == "grounded":
        return answer, False, None

    if grounding_status.get("status") == "no_tool_call":
        return (
            "I could not verify this answer against the knowledge base because the agent did not use the retrieval tool. "
            "Please retry or inspect the tool trace before trusting a generated answer.",
            True,
            grounding_status.get("message"),
        )

    return (
        "I could not find grounded evidence in the knowledge base for this question, so I am not returning a factual answer. "
        "Please refine the query or inspect the retrieved chunks.",
        True,
        grounding_status.get("message"),
    )


def _build_messages(question: str, history: list[dict[str, Any]] | None) -> list[BaseMessage]:
    messages: list[BaseMessage] = []
    assistant_messages = 0
    assistant_reasoning_messages = 0
    for item in history or []:
        role = item.get("role", "user")
        content = item.get("content", "")
        reasoning_content = item.get("reasoning_content")
        if not content:
            continue
        if role == "assistant":
            assistant_messages += 1
            additional_kwargs = {}
            if reasoning_content:
                additional_kwargs["reasoning_content"] = reasoning_content
                assistant_reasoning_messages += 1
            messages.append(AIMessage(content=content, additional_kwargs=additional_kwargs))
        elif role == "system":
            messages.append(SystemMessage(content=content))
        else:
            messages.append(HumanMessage(content=content))
    messages.append(HumanMessage(content=question))
    logger.info(
        "[react_agent.messages] history=%s total_messages=%s assistant_messages=%s assistant_reasoning=%s question_chars=%s",
        len(history or []),
        len(messages),
        assistant_messages,
        assistant_reasoning_messages,
        len(question),
    )
    return messages


def _serialize_tool_input(args: dict[str, Any]) -> str:
    try:
        return json.dumps(args, ensure_ascii=False)
    except TypeError:
        return str(args)


def _extract_answer(messages: list[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, AIMessage) and not getattr(message, "tool_calls", None):
            text = _stringify_content(message.content)
            if text:
                return text
    return ""


def _extract_reasoning_content(messages: list[BaseMessage]) -> str | None:
    for message in reversed(messages):
        if isinstance(message, AIMessage) and not getattr(message, "tool_calls", None):
            reasoning_content = message.additional_kwargs.get("reasoning_content")
            if reasoning_content:
                return str(reasoning_content)
    return None


def _build_agent_runtime(collection_name: str | None = None):
    documents: list[Document] = []
    resolved_collection = collection_name or settings.chroma_collection_name
    logger.info("[react_agent.runtime] build collection=%s", resolved_collection)

    @tool
    def knowledge_base_search(query: str) -> str:
        """Search the indexed knowledge base for evidence relevant to the user's question."""
        nonlocal documents

        logger.info(
            "[react_agent.tool.search] collection=%s query=%s",
            resolved_collection,
            query[:120],
        )
        vectorstore = get_vectorstore(collection_name)
        scored_docs = vectorstore.similarity_search_with_relevance_scores(
            query=query,
            k=settings.retriever_top_k,
            filter={"chunk_level": "child"},
        )
        tool_docs: list[Document] = []
        parent_chunk_ids: list[str] = []
        for doc, score in scored_docs:
            metadata = dict(doc.metadata)
            metadata["retrieval_score"] = round(float(score), 4)
            metadata["retrieval_hop"] = 1
            tool_docs.append(Document(page_content=doc.page_content, metadata=metadata))
            parent_chunk_id = str(metadata.get("parent_chunk_id") or "").strip()
            if parent_chunk_id:
                parent_chunk_ids.append(parent_chunk_id)

        parent_docs = get_parent_chunks_by_ids(parent_chunk_ids, collection_name)
        parent_by_id = {
            str(doc.metadata.get("parent_chunk_id") or ""): doc
            for doc in parent_docs
            if doc.metadata.get("parent_chunk_id")
        }
        grouped_children: dict[str, list[Document]] = {}
        for tool_doc in tool_docs:
            parent_chunk_id = str(tool_doc.metadata.get("parent_chunk_id") or "")
            grouped_children.setdefault(parent_chunk_id, []).append(tool_doc)

        enriched_tool_docs: list[Document] = []
        for tool_doc in tool_docs:
            metadata = dict(tool_doc.metadata)
            parent_chunk_id = str(metadata.get("parent_chunk_id") or "")
            parent_doc = parent_by_id.get(parent_chunk_id)
            if parent_doc is not None:
                metadata["parent_title"] = parent_doc.metadata.get("title") or parent_doc.metadata.get("section_path")
                metadata["parent_section_path"] = parent_doc.metadata.get("section_path") or parent_doc.metadata.get("parent_section_path")
                metadata["parent_content_preview"] = (
                    parent_doc.metadata.get("content_preview") or parent_doc.page_content
                )[: settings.rewrite_context_chars]
                metadata["matched_child_count"] = len(grouped_children.get(parent_chunk_id, []))
            enriched_tool_docs.append(Document(page_content=tool_doc.page_content, metadata=metadata))

        retrieval_audit = _build_retrieval_audit_text(enriched_tool_docs, parent_docs)
        documents = _merge_documents(documents, enriched_tool_docs)
        logger.info(
            "[react_agent.tool.search] complete results=%s parent_results=%s merged_documents=%s\n%s",
            len(enriched_tool_docs),
            len(parent_docs),
            len(documents),
            retrieval_audit,
        )
        return retrieval_audit + "\n\n" + _format_parent_child_documents(enriched_tool_docs, parent_docs)

    @tool
    def collection_overview() -> str:
        """Return internal metadata about the configured models and available collections."""
        logger.info("[react_agent.tool.meta] collection=%s", resolved_collection)
        collections = list_collections()
        summary = ", ".join(f"{item['name']}({item['count']})" for item in collections[:10]) or "none"
        return (
            f"Active LLM model: {settings.llm_model}\n"
            f"Active embedding model: {settings.embedding_model}\n"
            f"Default collection: {settings.chroma_collection_name}\n"
            f"Requested collection: {resolved_collection}\n"
            f"Collections: {summary}"
        )

    prompt = (
        "You are a ReAct-style question answering agent for a RAG system. "
        "When the user asks about indexed documents or knowledge-base content, use the knowledge_base_search tool before answering. "
        "Use collection_overview only for system metadata questions such as collection names, model configuration, or service state. "
        "Base your final answer only on tool results. If the tools do not provide enough information, say so directly. "
        "Keep answers concise and grounded."
    )
    agent = create_react_agent(
        model=_get_llm(),
        tools=[knowledge_base_search, collection_overview],
        prompt=prompt,
        name="rag_react_agent",
    )
    logger.info("[react_agent.runtime] ready tools=%s", 2)
    return agent, documents


def _build_react_agent_result(
    question: str,
    documents: list[Document],
    tool_calls: list[dict[str, str]],
    trace: list[str],
    answer: str,
    reasoning_content: str | None = None,
    debug_events: list[dict[str, Any]] | None = None,
    needs_human_review: bool = False,
    human_review_reason: str | None = None,
) -> dict[str, Any]:
    confidence_score = min(0.95, 0.35 + (0.15 * len(documents))) if documents else 0.2
    validation_issues = [] if documents else ["No grounded source chunks were returned by the agent tools."]
    logger.info(
        "[react_agent.result] docs=%s tool_calls=%s answer_chars=%s reasoning=%s confidence=%.2f",
        len(documents),
        len(tool_calls),
        len(answer),
        bool(reasoning_content),
        confidence_score,
    )
    return {
        "question": question,
        "answer": answer,
        "reasoning_content": reasoning_content,
        "question_type": "document_qa",
        "route": "react_agent",
        "plan": [
            "Decide whether a tool is needed",
            "Use knowledge base or metadata tools",
            "Synthesize a grounded final answer",
        ],
        "tool_calls": tool_calls,
        "documents": documents,
        "trace": trace,
        "debug_events": debug_events or [],
        "confidence_score": round(confidence_score, 4),
        "needs_human_review": needs_human_review,
        "human_review_reason": human_review_reason,
        "validation": {
            "passed": bool(answer.strip()) and bool(documents),
            "confidence": round(confidence_score, 4),
            "citations_verified": bool(documents),
            "issues": validation_issues,
        },
    }


async def run_react_agent_query(
    question: str,
    collection_name: str | None = None,
    history: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Answer a question with a ReAct-style agent loop backed by knowledge-base tools."""
    trace = ["1. Query accepted by React Agent API"]
    debug_events: list[dict[str, Any]] = []
    _append_debug_event(
        debug_events,
        "request_received",
        "Accepted React Agent request",
        collection=collection_name or settings.chroma_collection_name,
        history_count=len(history or []),
    )
    logger.info(
        "[react_agent.query] start collection=%s history=%s question=%s",
        collection_name or settings.chroma_collection_name,
        len(history or []),
        question[:120],
    )
    agent, documents = _build_agent_runtime(collection_name)
    _append_debug_event(
        debug_events,
        "runtime_ready",
        "Agent runtime initialized",
        tools=["knowledge_base_search", "collection_overview"],
    )

    model_input = {"messages": _build_messages(question, history)}
    _append_debug_event(
        debug_events,
        "model_cycle_start",
        "Starting agent reasoning loop",
        input_messages=len(model_input["messages"]),
    )
    result = await agent.ainvoke(model_input)
    messages = result.get("messages", [])
    logger.info("[react_agent.query] agent_returned_messages=%s", len(messages))
    _append_debug_event(
        debug_events,
        "model_cycle_complete",
        "Agent reasoning loop finished",
        returned_messages=len(messages),
    )

    tool_calls: list[dict[str, str]] = []
    pending_calls: dict[str, dict[str, str]] = {}
    trace_index = len(trace) + 1
    round_number = 0

    for message in messages:
        if isinstance(message, AIMessage) and getattr(message, "tool_calls", None):
            for call in message.tool_calls:
                round_number += 1
                call_id = call.get("id") or call.get("name") or f"tool-{len(pending_calls) + 1}"
                pending_calls[call_id] = {
                    "name": str(call.get("name", "unknown_tool")),
                    "status": "pending",
                    "input_summary": _serialize_tool_input(call.get("args", {})),
                    "output_summary": "",
                    "round_number": str(round_number),
                }
                _append_debug_event(
                    debug_events,
                    "thought",
                    f"Round {round_number}: model decided to call {pending_calls[call_id]['name']}",
                    round_number=round_number,
                    tool_name=pending_calls[call_id]["name"],
                    tool_input=pending_calls[call_id]["input_summary"],
                )
                _append_debug_event(
                    debug_events,
                    "action",
                    f"Round {round_number}: executing tool {pending_calls[call_id]['name']}",
                    round_number=round_number,
                    tool_name=pending_calls[call_id]["name"],
                )
                _log_react_stage(
                    "thought",
                    round_number,
                    f"selected tool={pending_calls[call_id]['name']} input={pending_calls[call_id]['input_summary']}",
                )
                trace.append(
                    f"{trace_index}. Agent selected tool {pending_calls[call_id]['name']}"
                )
                trace_index += 1
        elif isinstance(message, ToolMessage):
            call_id = message.tool_call_id or f"tool-{len(tool_calls) + 1}"
            entry = pending_calls.pop(
                call_id,
                {
                    "name": getattr(message, "name", "unknown_tool") or "unknown_tool",
                    "status": "success",
                    "input_summary": "",
                    "output_summary": "",
                },
            )
            entry["status"] = "success"
            entry["output_summary"] = _stringify_content(message.content)[:1600]
            tool_calls.append(entry)
            observe_round = int(entry.get("round_number", "0") or 0) or None
            _append_debug_event(
                debug_events,
                "observation",
                (
                    f"Round {observe_round}: tool {entry['name']} returned context"
                    if observe_round is not None
                    else f"Tool {entry['name']} returned context"
                ),
                round_number=observe_round,
                tool_name=entry["name"],
                output_preview=entry["output_summary"],
            )
            _log_react_stage(
                "observation",
                observe_round,
                f"tool={entry['name']} output_preview={entry['output_summary']}",
            )
            trace.append(f"{trace_index}. Tool {entry['name']} returned a result")
            trace_index += 1

    tool_calls.extend(
        {
            **entry,
            "status": "warning",
            "output_summary": entry["output_summary"] or "Tool call did not return a captured result.",
        }
        for entry in pending_calls.values()
    )

    answer = _extract_answer(messages)
    if not answer:
        answer = "I could not produce a grounded answer from the available tool results."
    reasoning_content = _extract_reasoning_content(messages)
    grounding_status = _build_grounding_status(documents, tool_calls)
    raw_answer = answer
    answer, needs_human_review, human_review_reason = _apply_grounding_guard(answer, grounding_status)
    grounding_context_text = _build_grounding_context_text(documents)
    grounding_context_details = _build_grounding_context_details(documents)
    logger.info("[react_agent.grounding.status] %s", grounding_status["message"])
    logger.info("[react_agent.grounding] documents=%s\n%s", len(documents), grounding_context_text)
    _append_debug_event(
        debug_events,
        "grounding_status",
        grounding_status["message"],
        status=grounding_status["status"],
        knowledge_search_calls=grounding_status["knowledge_search_calls"],
        metadata_calls=grounding_status["metadata_calls"],
        documents=grounding_status["documents"],
    )
    if answer != raw_answer:
        _append_debug_event(
            debug_events,
            "grounding_guard",
            "Replaced ungrounded model answer with a guarded fallback response",
            original_answer_chars=len(raw_answer),
            guarded_answer_chars=len(answer),
            reason=human_review_reason,
        )
    _append_debug_event(
        debug_events,
        "grounding_context",
        "Prepared final grounding context passed into answer synthesis",
        documents=len(documents),
        grounding_groups=len(grounding_context_details),
        grounding_preview=grounding_context_text[:2000],
        grounded_sources=grounding_context_details,
    )
    _append_debug_event(
        debug_events,
        "final",
        "Prepared final answer from agent messages",
        answer_chars=len(answer),
        reasoning_present=bool(reasoning_content),
        documents=len(documents),
    )
    _log_react_stage("final", None, f"answer_chars={len(answer)} documents={len(documents)}")

    trace.append(f"{len(trace) + 1}. React Agent generated the final answer")
    logger.info(
        "[react_agent.query] complete tool_calls=%s documents=%s reasoning=%s",
        len(tool_calls),
        len(documents),
        bool(reasoning_content),
    )
    return _build_react_agent_result(
        question,
        documents,
        tool_calls,
        trace,
        answer,
        reasoning_content=reasoning_content,
        debug_events=debug_events,
        needs_human_review=needs_human_review,
        human_review_reason=human_review_reason,
    )


def _extract_chunk_text(chunk: Any) -> str:
    content = getattr(chunk, "content", "")
    return _stringify_content(content)


async def stream_react_agent_query(
    question: str,
    collection_name: str | None = None,
    history: list[dict[str, str]] | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Stream React Agent events and emit a final structured payload."""
    trace = ["1. Query accepted by React Agent stream API"]
    debug_events: list[dict[str, Any]] = []
    _append_debug_event(
        debug_events,
        "request_received",
        "Accepted React Agent streaming request",
        collection=collection_name or settings.chroma_collection_name,
        history_count=len(history or []),
    )
    logger.info(
        "[react_agent.stream] start collection=%s history=%s question=%s",
        collection_name or settings.chroma_collection_name,
        len(history or []),
        question[:120],
    )
    yield {"type": "trace", "data": trace[-1]}
    yield {"type": "debug", "data": debug_events[-1]}

    agent, documents = _build_agent_runtime(collection_name)
    _append_debug_event(
        debug_events,
        "runtime_ready",
        "Agent runtime initialized",
        tools=["knowledge_base_search", "collection_overview"],
    )
    yield {"type": "debug", "data": debug_events[-1]}
    pending_calls: dict[str, dict[str, str]] = {}
    tool_calls: list[dict[str, str]] = []
    final_messages: list[BaseMessage] = []
    answer_tokens: list[str] = []
    emitted_answer_tokens = 0
    suppressed_answer_tokens = 0
    input_messages = _build_messages(question, history)
    round_number = 0

    _append_debug_event(
        debug_events,
        "model_cycle_start",
        "Starting agent reasoning loop",
        input_messages=len(input_messages),
    )
    yield {"type": "debug", "data": debug_events[-1]}

    async for event in agent.astream_events(
        {"messages": input_messages},
        version="v2",
    ):
        event_name = event.get("event", "")
        data = event.get("data", {}) or {}

        if event_name == "on_chat_model_stream":
            token = _extract_chunk_text(data.get("chunk"))
            if token:
                answer_tokens.append(token)
                if documents:
                    emitted_answer_tokens += 1
                    yield {"type": "token", "data": token}
                else:
                    suppressed_answer_tokens += 1
            continue

        if event_name == "on_tool_start":
            tool_name = str(event.get("name") or "unknown_tool")
            call_id = str(event.get("run_id") or f"tool-{len(pending_calls) + 1}")
            round_number += 1
            entry = {
                "name": tool_name,
                "status": "pending",
                "input_summary": _serialize_tool_input(data.get("input", {})),
                "output_summary": "",
                "round_number": str(round_number),
            }
            pending_calls[call_id] = entry
            logger.info("[react_agent.stream] tool_start name=%s pending=%s", tool_name, len(pending_calls))
            _append_debug_event(
                debug_events,
                "thought",
                f"Round {round_number}: model decided to call {tool_name}",
                round_number=round_number,
                tool_name=tool_name,
                tool_input=entry["input_summary"],
            )
            yield {"type": "debug", "data": debug_events[-1]}
            _append_debug_event(
                debug_events,
                "action",
                f"Round {round_number}: executing tool {tool_name}",
                round_number=round_number,
                tool_name=tool_name,
            )
            yield {"type": "debug", "data": debug_events[-1]}
            _log_react_stage(
                "thought",
                round_number,
                f"selected tool={tool_name} input={entry['input_summary']}",
            )
            trace_item = f"{len(trace) + 1}. Agent selected tool {tool_name}"
            trace.append(trace_item)
            yield {"type": "trace", "data": trace_item}
            continue

        if event_name == "on_tool_end":
            tool_name = str(event.get("name") or "unknown_tool")
            output_summary = _stringify_content(data.get("output"))[:1600]
            matching_id = None
            for call_id, entry in pending_calls.items():
                if entry["name"] == tool_name:
                    matching_id = call_id
                    break
            entry = pending_calls.pop(
                matching_id,
                {
                    "name": tool_name,
                    "status": "success",
                    "input_summary": "",
                    "output_summary": "",
                },
            )
            entry["status"] = "success"
            entry["output_summary"] = output_summary
            tool_calls.append(entry)
            logger.info("[react_agent.stream] tool_end name=%s completed=%s", tool_name, len(tool_calls))
            observe_round = int(entry.get("round_number", "0") or 0) or None
            _append_debug_event(
                debug_events,
                "observation",
                (
                    f"Round {observe_round}: tool {tool_name} returned context"
                    if observe_round is not None
                    else f"Tool {tool_name} returned context"
                ),
                round_number=observe_round,
                tool_name=tool_name,
                output_preview=output_summary,
            )
            yield {"type": "debug", "data": debug_events[-1]}
            _log_react_stage(
                "observation",
                observe_round,
                f"tool={tool_name} output_preview={output_summary}",
            )
            trace_item = f"{len(trace) + 1}. Tool {tool_name} returned a result"
            trace.append(trace_item)
            yield {"type": "trace", "data": trace_item}
            continue

        if event_name == "on_chain_end" and event.get("name") == "rag_react_agent":
            output = data.get("output")
            if isinstance(output, dict):
                final_messages = output.get("messages", []) or []
                logger.info("[react_agent.stream] chain_end messages=%s", len(final_messages))
                _append_debug_event(
                    debug_events,
                    "model_cycle_complete",
                    "Agent reasoning loop finished",
                    returned_messages=len(final_messages),
                )
                yield {"type": "debug", "data": debug_events[-1]}

    tool_calls.extend(
        {
            **entry,
            "status": "warning",
            "output_summary": entry["output_summary"] or "Tool call did not return a captured result.",
        }
        for entry in pending_calls.values()
    )

    answer = _extract_answer(final_messages) or "".join(answer_tokens).strip()
    if not answer:
        answer = "I could not produce a grounded answer from the available tool results."
    reasoning_content = _extract_reasoning_content(final_messages)
    grounding_status = _build_grounding_status(documents, tool_calls)
    raw_answer = answer
    answer, needs_human_review, human_review_reason = _apply_grounding_guard(answer, grounding_status)
    grounding_context_text = _build_grounding_context_text(documents)
    grounding_context_details = _build_grounding_context_details(documents)
    logger.info("[react_agent.grounding.status] %s", grounding_status["message"])
    logger.info("[react_agent.grounding] documents=%s\n%s", len(documents), grounding_context_text)
    if suppressed_answer_tokens:
        logger.info(
            "[react_agent.stream] suppressed_answer_tokens=%s emitted_answer_tokens=%s documents=%s",
            suppressed_answer_tokens,
            emitted_answer_tokens,
            len(documents),
        )
    _append_debug_event(
        debug_events,
        "grounding_status",
        grounding_status["message"],
        status=grounding_status["status"],
        knowledge_search_calls=grounding_status["knowledge_search_calls"],
        metadata_calls=grounding_status["metadata_calls"],
        documents=grounding_status["documents"],
    )
    yield {"type": "debug", "data": debug_events[-1]}
    if answer != raw_answer:
        _append_debug_event(
            debug_events,
            "grounding_guard",
            "Replaced ungrounded model answer with a guarded fallback response",
            original_answer_chars=len(raw_answer),
            guarded_answer_chars=len(answer),
            reason=human_review_reason,
        )
        yield {"type": "debug", "data": debug_events[-1]}
    _append_debug_event(
        debug_events,
        "grounding_context",
        "Prepared final grounding context passed into answer synthesis",
        documents=len(documents),
        grounding_groups=len(grounding_context_details),
        grounding_preview=grounding_context_text[:2000],
        grounded_sources=grounding_context_details,
    )
    yield {"type": "debug", "data": debug_events[-1]}
    _append_debug_event(
        debug_events,
        "final",
        "Prepared final answer from streamed agent messages",
        answer_chars=len(answer),
        reasoning_present=bool(reasoning_content),
        documents=len(documents),
        streamed_tokens=len(answer_tokens),
        emitted_tokens=emitted_answer_tokens,
        suppressed_tokens=suppressed_answer_tokens,
    )
    yield {"type": "debug", "data": debug_events[-1]}
    _log_react_stage("final", None, f"answer_chars={len(answer)} documents={len(documents)}")

    trace_item = f"{len(trace) + 1}. React Agent generated the final answer"
    trace.append(trace_item)
    logger.info(
        "[react_agent.stream] complete tool_calls=%s documents=%s tokens=%s emitted_tokens=%s suppressed_tokens=%s reasoning=%s",
        len(tool_calls),
        len(documents),
        len(answer_tokens),
        emitted_answer_tokens,
        suppressed_answer_tokens,
        bool(reasoning_content),
    )
    yield {"type": "trace", "data": trace_item}
    yield {
        "type": "final",
        "data": _build_react_agent_result(
            question,
            documents,
            tool_calls,
            trace,
            answer,
            reasoning_content=reasoning_content,
            debug_events=debug_events,
            needs_human_review=needs_human_review,
            human_review_reason=human_review_reason,
        ),
    }