from __future__ import annotations

import json
from typing import Any

import requests
import streamlit as st


DEFAULT_BACKEND_URL = "http://127.0.0.1:8004/api/v1"
REQUEST_TIMEOUT_SECONDS = 300


def _get_collection_name(selected_collection: str, custom_collection: str) -> str | None:
    custom_name = custom_collection.strip()
    if custom_name:
        return custom_name
    if selected_collection == "(default collection)":
        return None
    return selected_collection


def _fetch_collections(base_url: str) -> list[str]:
    response = requests.get(
        f"{base_url}/collections",
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    payload = response.json()
    return [item["name"] for item in payload.get("collections", [])]


def _upload_file(base_url: str, file_name: str, file_bytes: bytes, collection_name: str | None) -> dict[str, Any]:
    files = {"file": (file_name, file_bytes)}
    data: dict[str, str] = {}
    if collection_name:
        data["collection_name"] = collection_name

    response = requests.post(
        f"{base_url}/documents/upload",
        files=files,
        data=data,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def _fetch_chunks(
    base_url: str,
    collection_name: str | None,
    limit: int,
    offset: int,
    source_filter: str | None = None,
) -> dict[str, Any]:
    params: dict[str, Any] = {"limit": limit, "offset": offset}
    if collection_name:
        params["collection_name"] = collection_name
    if source_filter:
        params["source_filter"] = source_filter

    response = requests.get(
        f"{base_url}/documents/chunks",
        params=params,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def _delete_chunks_by_source(
    base_url: str,
    source: str,
    collection_name: str | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"source": source}
    if collection_name:
        payload["collection_name"] = collection_name

    response = requests.request(
        "DELETE",
        f"{base_url}/documents/source/delete",
        json=payload,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def _query(base_url: str, question: str, collection_name: str | None) -> dict[str, Any]:
    payload: dict[str, Any] = {"question": question}
    if collection_name:
        payload["collection_name"] = collection_name

    response = requests.post(
        f"{base_url}/query",
        json=payload,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def _query_react_agent(
    base_url: str,
    question: str,
    collection_name: str | None,
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    payload: dict[str, Any] = {"question": question, "history": history}
    if collection_name:
        payload["collection_name"] = collection_name

    response = requests.post(
        f"{base_url}/chat/react-agent",
        json=payload,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def _stream_react_agent(
    base_url: str,
    question: str,
    collection_name: str | None,
    history: list[dict[str, Any]],
):
    payload: dict[str, Any] = {"question": question, "history": history}
    if collection_name:
        payload["collection_name"] = collection_name

    with requests.post(
        f"{base_url}/chat/react-agent/stream",
        json=payload,
        timeout=REQUEST_TIMEOUT_SECONDS,
        stream=True,
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            yield json.loads(line)


def _session_key(chat_mode: str) -> str:
    return "react_agent" if chat_mode == "React Agent" else "langgraph"


def _build_history(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    history: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        content = message.get("content", "")
        if role in {"user", "assistant", "system"} and content:
            payload = {"role": role, "content": content}
            reasoning_content = message.get("reasoning_content")
            if reasoning_content:
                payload["reasoning_content"] = reasoning_content
            history.append(payload)
    return history


def _render_history_preview(messages: list[dict[str, Any]]) -> None:
    history = _build_history(messages)
    st.caption(f"History messages sent on next turn: {len(history)}")
    if not history:
        st.caption("No history will be sent to the model.")
        return

    with st.expander("Current model history", expanded=False):
        for index, item in enumerate(history[-6:], start=max(1, len(history) - 5)):
            role = item.get("role", "unknown")
            content = str(item.get("content", "")).strip()
            preview = content if len(content) <= 220 else f"{content[:220]}..."
            st.markdown(f"**{index}. {role}**")
            st.write(preview or "(empty)")
            if item.get("reasoning_content"):
                st.caption("Includes hidden reasoning_content")


def _render_reasoning_content(reasoning_content: str | None) -> None:
    if not reasoning_content:
        return

    with st.expander("LLM reasoning", expanded=False):
        st.text(reasoning_content)


def _render_debug_panel(events: list[dict[str, Any]]) -> None:
    with st.expander("Agent debug panel", expanded=False):
        if not events:
            st.caption("No debug events captured for this answer.")
            return
        phase_label = {
            "request_received": "Request",
            "runtime_ready": "Setup",
            "model_cycle_start": "Loop Start",
            "model_cycle_complete": "Loop End",
            "thought": "Thought",
            "action": "Action",
            "observation": "Observation",
            "final": "Final",
        }
        for index, event in enumerate(events, start=1):
            phase = event.get("phase", "unknown")
            message = event.get("message", "")
            details = event.get("details") or {}
            round_number = details.get("round_number")
            round_prefix = f"Round {round_number}" if round_number else f"Step {index}"
            label = phase_label.get(phase, phase.replace("_", " ").title())
            st.markdown(f"**{round_prefix} · {label}**")
            st.caption(message)
            if details:
                summary_parts = []
                if details.get("tool_name"):
                    summary_parts.append(f"tool={details['tool_name']}")
                if details.get("tool_input"):
                    summary_parts.append(f"input={details['tool_input']}")
                if details.get("output_preview"):
                    summary_parts.append(f"observation={details['output_preview']}")
                if details.get("input_messages"):
                    summary_parts.append(f"messages={details['input_messages']}")
                if details.get("returned_messages"):
                    summary_parts.append(f"returned={details['returned_messages']}")
                if details.get("documents") is not None:
                    summary_parts.append(f"documents={details['documents']}")
                if details.get("answer_chars") is not None:
                    summary_parts.append(f"answer_chars={details['answer_chars']}")
                if summary_parts:
                    st.code(" | ".join(summary_parts), language=None)
                with st.container(border=True):
                    st.json(details)


def _init_state() -> None:
    if "chat_sessions" not in st.session_state:
        st.session_state.chat_sessions = {"langgraph": [], "react_agent": []}
    if "collection_names" not in st.session_state:
        st.session_state.collection_names = []
    if "chunk_browser" not in st.session_state:
        st.session_state.chunk_browser = {
            "offset": 0,
            "limit": 20,
            "source_filter": "",
            "payload": None,
        }


def _render_agentic_details(result: dict[str, Any]) -> None:
    metadata_cols = st.columns(4)
    metadata_cols[0].metric("Question type", result.get("question_type", "unknown"))
    metadata_cols[1].metric("Route", result.get("route", "unknown"))
    confidence = result.get("confidence_score")
    metadata_cols[2].metric(
        "Confidence",
        f"{float(confidence):.2f}" if confidence is not None else "n/a",
    )
    metadata_cols[3].metric(
        "Human review",
        "required" if result.get("needs_human_review") else "not needed",
    )

    if result.get("needs_human_review"):
        st.warning(result.get("human_review_reason") or "Human review required for this answer.")

    if result.get("plan"):
        with st.expander("Execution plan", expanded=False):
            for index, step in enumerate(result["plan"], start=1):
                st.write(f"{index}. {step}")

    if result.get("tool_calls"):
        with st.expander("Tool calls", expanded=False):
            for call in result["tool_calls"]:
                st.markdown(f"**{call.get('name', 'unknown tool')}** [{call.get('status', 'unknown')}]")
                if call.get("input_summary"):
                    st.caption(f"Input: {call['input_summary']}")
                if call.get("output_summary"):
                    st.write(call["output_summary"])

    validation = result.get("validation") or {}
    if validation:
        with st.expander("Validation report", expanded=False):
            st.write(f"Passed: {validation.get('passed', False)}")
            st.write(f"Citations verified: {validation.get('citations_verified', False)}")
            if validation.get("issues"):
                for issue in validation["issues"]:
                    st.write(f"- {issue}")


def _render_sources(sources: list[dict[str, Any]]) -> None:
    with st.expander("Retrieved sources", expanded=False):
        for source in sources:
            title = source.get("source", "unknown source")
            retrieval_score = source.get("retrieval_score")
            retrieval_hop = source.get("retrieval_hop")
            summary = [title]
            if source.get("title"):
                summary.append(str(source["title"]))
            if source.get("chunk_level"):
                summary.append(str(source["chunk_level"]))
            if retrieval_hop is not None:
                summary.append(f"hop {retrieval_hop}")
            if retrieval_score is not None:
                summary.append(f"score {float(retrieval_score):.4f}")
            st.markdown(f"**{' | '.join(summary)}**")
            if source.get("section_path"):
                st.caption(f"Section path: {source['section_path']}")
            if source.get("parent_chunk_id"):
                st.caption(f"Parent chunk: {source['parent_chunk_id']}")
            with st.container(border=True):
                st.caption("Matched child evidence")
                st.write(source.get("content", ""))
            if source.get("parent_content"):
                with st.container(border=True):
                    parent_label = source.get("parent_title") or source.get("parent_section_path") or "parent context"
                    st.caption(f"Parent context used by agent: {parent_label}")
                    if source.get("parent_section_path"):
                        st.caption(f"Parent section: {source['parent_section_path']}")
                    st.write(source.get("parent_content", ""))


def _render_chunks(payload: dict[str, Any]) -> None:
    st.caption(
        f"Collection: {payload.get('collection_name', 'unknown')} | "
        f"Source filter: {payload.get('source_filter') or 'all'} | "
        f"Showing {len(payload.get('chunks', []))} / {payload.get('total', 0)} chunks"
    )
    for index, chunk in enumerate(payload.get("chunks", []), start=1):
        title_parts = [f"#{index + int(payload.get('offset', 0))}", chunk.get("source") or "unknown source"]
        if chunk.get("title"):
            title_parts.append(str(chunk["title"]))
        if chunk.get("chunk_level"):
            title_parts.append(str(chunk["chunk_level"]))
        with st.expander(" | ".join(title_parts), expanded=False):
            meta_cols = st.columns(3)
            meta_cols[0].caption(f"Page: {chunk.get('page') or 'n/a'}")
            meta_cols[1].caption(f"Section: {chunk.get('section_path') or 'root'}")
            meta_cols[2].caption(f"Chunk ID: {chunk.get('id')}")
            st.json(
                {
                    "source_type": chunk.get("source_type"),
                    "chunk_level": chunk.get("chunk_level"),
                    "title": chunk.get("title"),
                    "tags": chunk.get("tags", []),
                    "document_id": chunk.get("document_id"),
                    "section_depth": chunk.get("section_depth"),
                    "chunk_size": chunk.get("chunk_size"),
                    "chunk_overlap": chunk.get("chunk_overlap"),
                    "has_children": chunk.get("has_children"),
                    "parent_section_path": chunk.get("parent_section_path"),
                    "parent_chunk_id": chunk.get("parent_chunk_id"),
                    "parent_chunk_index": chunk.get("parent_chunk_index"),
                    "child_chunk_index": chunk.get("child_chunk_index"),
                    "child_chunk_count": chunk.get("child_chunk_count"),
                    "child_chunk_start_index": chunk.get("child_chunk_start_index"),
                    "child_chunk_end_index": chunk.get("child_chunk_end_index"),
                    "start_index": chunk.get("start_index"),
                }
            )
            st.text_area(
                "Chunk content",
                value=chunk.get("content", ""),
                height=220,
                disabled=True,
                key=f"chunk-content-{chunk.get('id')}",
                label_visibility="collapsed",
            )


st.set_page_config(page_title="ragEnlighten UI", page_icon="📚", layout="wide")
_init_state()

st.title("ragEnlighten")
st.caption("Upload files into the knowledge base and ask grounded questions against the indexed content.")


with st.sidebar:
    st.header("Backend")
    backend_url = st.text_input("FastAPI base URL", value=DEFAULT_BACKEND_URL)
    chat_mode = st.radio(
        "Chat mode",
        options=["LangGraph", "React Agent"],
        captions=[
            "Use the existing LangGraph pipeline",
            "Use the new React Agent endpoint with streaming output",
        ],
    )
    if st.button("Refresh collections", use_container_width=True):
        try:
            st.session_state.collection_names = _fetch_collections(backend_url)
            st.success("Collections refreshed")
        except requests.RequestException as exc:
            st.error(f"Failed to load collections: {exc}")

    active_session_key = _session_key(chat_mode)
    active_chat_session = st.session_state.chat_sessions[active_session_key]

    collection_options = ["(default collection)", *st.session_state.collection_names]
    selected_collection = st.selectbox("Existing collection", options=collection_options)
    custom_collection = st.text_input("Or enter a collection name", value="")
    active_collection = _get_collection_name(selected_collection, custom_collection)

    st.divider()
    st.caption("Current target collection")
    st.code(active_collection or "rag_documents", language=None)

    if chat_mode == "React Agent":
        st.divider()
        st.subheader("React Agent Session")
        st.caption(f"Current messages in session: {len(active_chat_session)}")
        if st.button("Clear React Agent Session", use_container_width=True):
            st.session_state.chat_sessions["react_agent"] = []
            st.rerun()
        _render_history_preview(active_chat_session)


upload_tab, chat_tab, chunks_tab = st.tabs(["Upload", "Chat", "Chunks"])


with upload_tab:
    st.subheader("Upload document")
    uploaded_file = st.file_uploader(
        "Choose a document",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=False,
    )
    if st.button("Upload and index", type="primary", use_container_width=True):
        if uploaded_file is None:
            st.warning("Choose a file before uploading.")
        else:
            try:
                with st.spinner("Uploading and indexing document..."):
                    result = _upload_file(
                        backend_url,
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        active_collection,
                    )
                st.success(result["message"])
                st.json(result)
                st.session_state.collection_names = _fetch_collections(backend_url)
            except requests.RequestException as exc:
                details = exc.response.text if exc.response is not None else str(exc)
                st.error(f"Upload failed: {details}")


with chat_tab:
    st.subheader("Ask the knowledge base")
    st.caption(
        "LangGraph mode uses `/query`. React Agent mode uses the streaming `/chat/react-agent/stream` endpoint."
    )

    current_session = st.session_state.chat_sessions[_session_key(chat_mode)]

    for message in current_session:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                _render_reasoning_content(message.get("reasoning_content"))
                if message.get("trace"):
                    with st.expander("Execution trace", expanded=True):
                        for item in message["trace"]:
                            st.write(item)
                if message.get("debug_events"):
                    _render_debug_panel(message["debug_events"])
                if message.get("agentic_details"):
                    _render_agentic_details(message["agentic_details"])
                if message.get("sources"):
                    _render_sources(message["sources"])

    question = st.chat_input("Ask a question about the indexed documents")
    if question:
        current_session.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                if chat_mode == "React Agent":
                    status = st.status("Streaming React Agent...", expanded=True)
                    status.write("Opening streaming connection to FastAPI backend")
                    answer_placeholder = st.empty()
                    streamed_answer = ""
                    final_result: dict[str, Any] | None = None
                    streamed_debug_events: list[dict[str, Any]] = []

                    for event in _stream_react_agent(
                        backend_url,
                        question,
                        active_collection,
                        _build_history(current_session[:-1]),
                    ):
                        event_type = event.get("type")
                        if event_type == "token":
                            streamed_answer += event.get("data", "")
                            answer_placeholder.markdown(streamed_answer or " ")
                        elif event_type == "debug":
                            debug_event = event.get("data", {})
                            streamed_debug_events.append(debug_event)
                            status.write(
                                f"DEBUG [{debug_event.get('phase', 'unknown')}]: {debug_event.get('message', '')}"
                            )
                        elif event_type == "trace":
                            status.write(event.get("data", ""))
                        elif event_type == "error":
                            raise requests.RequestException(event.get("data", "Unknown stream error"))
                        elif event_type == "final":
                            final_result = event.get("data", {})

                    if final_result is None:
                        raise requests.RequestException("React Agent stream ended without a final payload.")

                    trace = final_result.get("trace", [])
                    answer_placeholder.markdown(final_result["answer"])
                    status_label = (
                        "Human review required"
                        if final_result.get("needs_human_review")
                        else "React Agent stream complete"
                    )
                    status_state = "error" if final_result.get("needs_human_review") else "complete"
                    status.update(label=status_label, state=status_state)
                    _render_reasoning_content(final_result.get("reasoning_content"))
                    _render_agentic_details(final_result)
                    if trace:
                        with st.expander("Execution trace", expanded=True):
                            for item in trace:
                                st.write(item)
                    final_debug_events = final_result.get("debug_events") or streamed_debug_events
                    if final_debug_events:
                        _render_debug_panel(final_debug_events)
                    if final_result.get("sources"):
                        _render_sources(final_result["sources"])

                    current_session.append(
                        {
                            "role": "assistant",
                            "content": final_result["answer"],
                            "reasoning_content": final_result.get("reasoning_content"),
                            "trace": trace,
                            "debug_events": final_debug_events,
                            "agentic_details": {
                                "question_type": final_result.get("question_type"),
                                "route": final_result.get("route"),
                                "plan": final_result.get("plan", []),
                                "tool_calls": final_result.get("tool_calls", []),
                                "confidence_score": final_result.get("confidence_score"),
                                "needs_human_review": final_result.get("needs_human_review", False),
                                "human_review_reason": final_result.get("human_review_reason"),
                                "validation": final_result.get("validation"),
                            },
                            "sources": final_result.get("sources", []),
                        }
                    )
                else:
                    status = st.status("Running RAG pipeline...", expanded=True)
                    status.write("Sending query to FastAPI backend")
                    result = _query(backend_url, question, active_collection)
                    trace = result.get("trace", [])
                    for item in trace:
                        status.write(item)
                    status_label = "Human review required" if result.get("needs_human_review") else "RAG pipeline complete"
                    status_state = "error" if result.get("needs_human_review") else "complete"
                    status.update(label=status_label, state=status_state)

                    st.markdown(result["answer"])
                    _render_reasoning_content(result.get("reasoning_content"))
                    _render_agentic_details(result)
                    if trace:
                        with st.expander("Execution trace", expanded=True):
                            for item in trace:
                                st.write(item)
                    if result.get("debug_events"):
                        _render_debug_panel(result["debug_events"])
                    if result.get("sources"):
                        _render_sources(result["sources"])

                    current_session.append(
                        {
                            "role": "assistant",
                            "content": result["answer"],
                            "trace": trace,
                            "debug_events": result.get("debug_events", []),
                            "agentic_details": {
                                "question_type": result.get("question_type"),
                                "route": result.get("route"),
                                "plan": result.get("plan", []),
                                "tool_calls": result.get("tool_calls", []),
                                "confidence_score": result.get("confidence_score"),
                                "needs_human_review": result.get("needs_human_review", False),
                                "human_review_reason": result.get("human_review_reason"),
                                "validation": result.get("validation"),
                            },
                            "sources": result.get("sources", []),
                        }
                    )
            except requests.RequestException as exc:
                details = exc.response.text if exc.response is not None else str(exc)
                status.update(label="Chat request failed", state="error")
                st.error(details)


with chunks_tab:
    st.subheader("Browse stored chunks")
    st.caption("Inspect the chunks currently stored in the selected vector collection.")
    browser_state = st.session_state.chunk_browser
    chunk_limit = st.number_input(
        "Chunks per page",
        min_value=1,
        max_value=100,
        value=int(browser_state.get("limit", 20)),
        step=5,
    )
    source_filter = st.text_input(
        "Filter by source",
        value=str(browser_state.get("source_filter", "")),
        placeholder="e.g. resume.pdf",
    )
    controls = st.columns(3)
    load_requested = controls[0].button("Load chunks", use_container_width=True)
    previous_requested = controls[1].button("Previous page", use_container_width=True)
    next_requested = controls[2].button("Next page", use_container_width=True)

    if previous_requested:
        browser_state["offset"] = max(0, int(browser_state.get("offset", 0)) - int(chunk_limit))
    elif next_requested:
        browser_state["offset"] = int(browser_state.get("offset", 0)) + int(chunk_limit)

    filter_changed = (
        int(browser_state.get("limit", 20)) != int(chunk_limit)
        or str(browser_state.get("source_filter", "")) != source_filter
    )
    if filter_changed:
        browser_state["offset"] = 0

    browser_state["limit"] = int(chunk_limit)
    browser_state["source_filter"] = source_filter

    if load_requested or previous_requested or next_requested or filter_changed:
        try:
            with st.spinner("Loading chunks from vector store..."):
                chunk_payload = _fetch_chunks(
                    backend_url,
                    active_collection,
                    int(browser_state["limit"]),
                    int(browser_state["offset"]),
                    browser_state["source_filter"].strip() or None,
                )
            total = int(chunk_payload.get("total", 0))
            current_offset = int(browser_state["offset"])
            if total and current_offset >= total:
                browser_state["offset"] = max(
                    0,
                    ((total - 1) // int(browser_state["limit"])) * int(browser_state["limit"]),
                )
                chunk_payload = _fetch_chunks(
                    backend_url,
                    active_collection,
                    int(browser_state["limit"]),
                    int(browser_state["offset"]),
                    browser_state["source_filter"].strip() or None,
                )
            browser_state["payload"] = chunk_payload
        except requests.RequestException as exc:
            details = exc.response.text if exc.response is not None else str(exc)
            st.error(f"Failed to load chunks: {details}")

    payload = browser_state.get("payload")
    if payload:
        nav_cols = st.columns(3)
        nav_cols[0].caption(f"Offset: {payload.get('offset', 0)}")
        nav_cols[1].caption(f"Page size: {payload.get('limit', 20)}")
        nav_cols[2].caption(f"Total: {payload.get('total', 0)}")

        active_source_filter = (browser_state.get("source_filter", "") or "").strip()
        delete_disabled = not active_source_filter
        delete_help = (
            "Enter a source filter before deleting vector data."
            if delete_disabled
            else f"Delete all chunks whose source is '{active_source_filter}'."
        )
        if st.button(
            "Delete current source data",
            type="secondary",
            use_container_width=True,
            disabled=delete_disabled,
            help=delete_help,
        ):
            try:
                with st.spinner("Deleting vector data from the selected collection..."):
                    delete_result = _delete_chunks_by_source(
                        backend_url,
                        active_source_filter,
                        active_collection,
                    )
                st.success(
                    f"Deleted {delete_result['deleted_count']} chunks for source '{delete_result['source']}'."
                )
                browser_state["offset"] = 0
                browser_state["payload"] = _fetch_chunks(
                    backend_url,
                    active_collection,
                    int(browser_state["limit"]),
                    0,
                    active_source_filter,
                )
                st.session_state.collection_names = _fetch_collections(backend_url)
            except requests.RequestException as exc:
                details = exc.response.text if exc.response is not None else str(exc)
                st.error(f"Delete failed: {details}")

        _render_chunks(payload)