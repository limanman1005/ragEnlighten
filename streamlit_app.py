from __future__ import annotations

from typing import Any

import requests
import streamlit as st


DEFAULT_BACKEND_URL = "http://127.0.0.1:8004/api/v1"
REQUEST_TIMEOUT_SECONDS = 120


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


def _init_state() -> None:
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "collection_names" not in st.session_state:
        st.session_state.collection_names = []


st.set_page_config(page_title="ragEnlighten UI", page_icon="📚", layout="wide")
_init_state()

st.title("ragEnlighten")
st.caption("Upload files into the knowledge base and ask grounded questions against the indexed content.")


with st.sidebar:
    st.header("Backend")
    backend_url = st.text_input("FastAPI base URL", value=DEFAULT_BACKEND_URL)
    if st.button("Refresh collections", use_container_width=True):
        try:
            st.session_state.collection_names = _fetch_collections(backend_url)
            st.success("Collections refreshed")
        except requests.RequestException as exc:
            st.error(f"Failed to load collections: {exc}")

    collection_options = ["(default collection)", *st.session_state.collection_names]
    selected_collection = st.selectbox("Existing collection", options=collection_options)
    custom_collection = st.text_input("Or enter a collection name", value="")
    active_collection = _get_collection_name(selected_collection, custom_collection)

    st.divider()
    st.caption("Current target collection")
    st.code(active_collection or "rag_documents", language=None)


upload_tab, chat_tab = st.tabs(["Upload", "Chat"])


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
    st.caption("The execution trace below each answer comes from the backend query response.")

    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                if message.get("trace"):
                    with st.expander("Execution trace", expanded=True):
                        for item in message["trace"]:
                            st.write(item)
                if message.get("sources"):
                    with st.expander("Retrieved sources", expanded=False):
                        for source in message["sources"]:
                            st.markdown(f"**{source.get('source', 'unknown source')}**")
                            st.write(source.get("content", ""))

    question = st.chat_input("Ask a question about the indexed documents")
    if question:
        st.session_state.chat_messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            status = st.status("Running RAG pipeline...", expanded=True)
            status.write("Sending query to FastAPI backend")
            try:
                result = _query(backend_url, question, active_collection)
                trace = result.get("trace", [])
                for item in trace:
                    status.write(item)
                status.update(label="RAG pipeline complete", state="complete")

                st.markdown(result["answer"])
                if trace:
                    with st.expander("Execution trace", expanded=True):
                        for item in trace:
                            st.write(item)
                if result.get("sources"):
                    with st.expander("Retrieved sources", expanded=False):
                        for source in result["sources"]:
                            st.markdown(f"**{source.get('source', 'unknown source')}**")
                            st.write(source.get("content", ""))

                st.session_state.chat_messages.append(
                    {
                        "role": "assistant",
                        "content": result["answer"],
                        "trace": trace,
                        "sources": result.get("sources", []),
                    }
                )
            except requests.RequestException as exc:
                details = exc.response.text if exc.response is not None else str(exc)
                status.update(label="RAG pipeline failed", state="error")
                st.error(details)