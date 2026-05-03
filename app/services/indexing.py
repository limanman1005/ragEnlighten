"""Document indexing service.

Handles loading, splitting, embedding and persisting documents into the
Chroma vector store.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import BinaryIO

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

from app.core.config import settings


# Supported MIME / extension -> loader mapping
_EXTENSION_LOADERS: dict[str, type] = {
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".txt": TextLoader,
    ".md": UnstructuredMarkdownLoader,
}


def _get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=settings.openai_embedding_model,
        openai_api_key=settings.openai_api_key,
    )


def get_vectorstore(collection_name: str | None = None) -> Chroma:
    """Return (or create) a Chroma vector store for the given collection."""
    name = collection_name or settings.chroma_collection_name
    return Chroma(
        collection_name=name,
        embedding_function=_get_embeddings(),
        persist_directory=settings.chroma_persist_dir,
    )


def index_file(
    file_content: bytes,
    filename: str,
    collection_name: str | None = None,
) -> int:
    """Persist *file_content* into the vector store and return the chunk count.

    Parameters
    ----------
    file_content:
        Raw bytes of the uploaded file.
    filename:
        Original filename (used to choose the loader and set metadata).
    collection_name:
        Target collection; falls back to ``settings.chroma_collection_name``.

    Returns
    -------
    int
        Number of chunks written to the vector store.
    """
    suffix = Path(filename).suffix.lower()
    loader_cls = _EXTENSION_LOADERS.get(suffix)
    if loader_cls is None:
        raise ValueError(
            f"Unsupported file type '{suffix}'. "
            f"Supported types: {list(_EXTENSION_LOADERS.keys())}"
        )

    # Write to a temp file so that document loaders can read it.
    # tempfile.mkstemp() creates the file with mode 0o600 (owner-only read/write)
    # from the very start on POSIX systems, so no chmod race window exists.
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as tmp:
            tmp.write(file_content)
    except Exception:
        os.unlink(tmp_path)
        raise

    try:
        loader = loader_cls(tmp_path)
        docs: list[Document] = loader.load()
    finally:
        os.unlink(tmp_path)

    # Attach the original filename as source metadata
    for doc in docs:
        doc.metadata.setdefault("source", filename)

    return _split_and_store(docs, collection_name)


def index_text(
    text: str,
    source: str = "inline",
    collection_name: str | None = None,
) -> int:
    """Index a raw text string and return the chunk count."""
    docs = [Document(page_content=text, metadata={"source": source})]
    return _split_and_store(docs, collection_name)


def _split_and_store(
    docs: list[Document],
    collection_name: str | None = None,
) -> int:
    """Split *docs* into chunks and persist them.  Returns chunk count."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )
    chunks = splitter.split_documents(docs)
    if not chunks:
        return 0

    vs = get_vectorstore(collection_name)
    vs.add_documents(chunks)
    return len(chunks)


def list_collections() -> list[dict[str, int | str]]:
    """Return all Chroma collections with their document counts."""
    import chromadb

    client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    result = []
    for col in client.list_collections():
        result.append({"name": col.name, "count": col.count()})
    return result


def delete_document(doc_id: str, collection_name: str | None = None) -> None:
    """Delete a document by its Chroma *doc_id*."""
    vs = get_vectorstore(collection_name)
    vs.delete([doc_id])
