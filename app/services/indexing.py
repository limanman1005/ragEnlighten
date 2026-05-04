"""Document indexing service.

Handles loading, splitting, embedding and persisting documents into the
Chroma vector store.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import tempfile
from pathlib import Path
from langchain_core.documents import Document

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain_community.embeddings.dashscope import DashScopeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

from app.core.config import settings


logger = logging.getLogger("uvicorn.error")


# Supported MIME / extension -> loader mapping
_EXTENSION_LOADERS: dict[str, type] = {
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".txt": TextLoader,
    ".md": UnstructuredMarkdownLoader,
}

_MARKDOWN_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
_CHAPTER_HEADING_RE = re.compile(r"^(第[\d一二三四五六七八九十百千]+[章节部篇卷]|[一二三四五六七八九十]+、)\s*(.*\S)?$")
_NUMBERED_HEADING_RE = re.compile(r"^(\d+(?:\.\d+){0,3})[\s、.．-]+(.*\S)$")
_YEAR_RANGE_HEADING_RE = re.compile(
    r"^((?:19|20)\d{2})(?:\s*[-/~至]\s*((?:19|20)\d{2}|至今|present|Present))?(?:\s*[|｜·•-]\s*.*)?$"
)
_SECTION_BREAK_RE = re.compile(r"\n(?:目录|contents)\b.*?(?=\n\S|$)", re.IGNORECASE | re.DOTALL)
_TAG_SPLIT_RE = re.compile(r"[\s/|｜·•,，;；:：()\[\]{}]+")
_CLAUSE_HEADING_RE = re.compile(
    r"^(第[\d一二三四五六七八九十百千]+(?:章|节|条|款|项)|[（(]?[\d一二三四五六七八九十]+[)）.、])\s*.+$"
)
_KEY_VALUE_LINE_RE = re.compile(r"^[^:：\n]{1,40}\s*[:：]\s*\S.+$")

_CHUNKING_PROFILES: dict[str, dict[str, object]] = {
    "md": {
        "separators": ["\n# ", "\n## ", "\n### ", "\n#### ", "\n\n", "\n", "。", ". ", " ", ""],
        "parent_size": 1800,
        "parent_overlap": 220,
        "child_size": 500,
        "child_overlap": 80,
    },
    "pdf": {
        "separators": ["\n\n", "\n", "。", "；", ". ", "; ", " ", ""],
        "parent_size": 1800,
        "parent_overlap": 240,
        "child_size": 200,
        "child_overlap": 50,
    },
    "docx": {
        "separators": ["\n\n", "\n", "。", ". ", "; ", " ", ""],
        "parent_size": 1600,
        "parent_overlap": 220,
        "child_size": 520,
        "child_overlap": 90,
    },
    "default": {
        "separators": ["\n\n", "\n", "。", ". ", " ", ""],
        "parent_size": 1400,
        "parent_overlap": 180,
        "child_size": 420,
        "child_overlap": 70,
    },
}

_STRUCTURE_CHUNKING_PROFILES: dict[str, dict[str, object]] = {
    "strong_structured": {
        "separators": ["\n# ", "\n## ", "\n### ", "\n\n", "\n第", "\n", "。", ". ", " ", ""],
        "parent_size": 2200,
        "parent_overlap": 260,
        "child_size": 650,
        "child_overlap": 120,
    },
    "weak_structured": {
        "separators": ["\n\n", "\n", "。", "；", ". ", "; ", " ", ""],
        "parent_size": 1500,
        "parent_overlap": 220,
        "child_size": 480,
        "child_overlap": 90,
    },
    "semi_structured": {
        "separators": ["\n\n", "\n", "；", "; ", "。", ". ", ""],
        "parent_size": 1400,
        "parent_overlap": 320,
        "child_size": 520,
        "child_overlap": 180,
    },
}


def _detect_source_type(metadata: dict[str, object]) -> str:
    source_type = str(metadata.get("source_type", "")).strip().lower()
    if source_type:
        return source_type
    source = str(metadata.get("source", "")).lower()
    suffix = Path(source).suffix.lower().lstrip(".")
    return suffix or "text"


def _is_table_like_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if stripped.count("|") >= 2 or "\t" in stripped:
        return True
    if _KEY_VALUE_LINE_RE.match(stripped):
        return True
    return bool(re.search(r"\S(?:\s{2,}|\t)\S(?:\s{2,}|\t)\S", stripped))


def _classify_document_structure(text: str, source_type: str) -> tuple[str, dict[str, int | float]]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    paragraphs = [block.strip() for block in re.split(r"\n{2,}", text) if block.strip()]
    heading_lines = sum(
        1 for line in lines if _infer_heading(line, source_type) is not None or _CLAUSE_HEADING_RE.match(line)
    )
    table_lines = sum(1 for line in lines if _is_table_like_line(line))
    key_value_lines = sum(1 for line in lines if _KEY_VALUE_LINE_RE.match(line))
    long_paragraphs = sum(1 for block in paragraphs if len(block) >= 220)
    total_lines = max(1, len(lines))
    total_paragraphs = max(1, len(paragraphs))

    if source_type == "md":
        structure = "strong_structured"
    elif table_lines >= max(3, total_lines // 3) or key_value_lines >= max(4, total_lines // 4):
        structure = "semi_structured"
    elif heading_lines >= 3 and heading_lines / total_lines >= 0.08:
        structure = "strong_structured"
    elif source_type in {"pdf", "docx"} and (table_lines + key_value_lines) >= 4:
        structure = "semi_structured"
    elif long_paragraphs >= max(2, total_paragraphs // 2):
        structure = "weak_structured"
    else:
        structure = "weak_structured"

    return structure, {
        "heading_lines": heading_lines,
        "table_lines": table_lines,
        "key_value_lines": key_value_lines,
        "paragraphs": len(paragraphs),
        "long_paragraphs": long_paragraphs,
        "nonempty_lines": len(lines),
    }


def _clean_text(text: str, source_type: str, document_structure: str | None = None) -> str:
    cleaned = text.replace("\x00", " ").replace("\ufeff", "")
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = cleaned.replace("\t", "    ")
    cleaned = _SECTION_BREAK_RE.sub("\n", cleaned)
    cleaned = re.sub(r"[ \u3000]+", " ", cleaned)
    cleaned = re.sub(r"\n[ \t]+", "\n", cleaned)
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    if source_type in {"pdf", "docx", "txt", "text"} and document_structure == "weak_structured":
        cleaned = re.sub(r"(?<![\n])\n(?=[^\n•\-\d#])", " ", cleaned)
        cleaned = re.sub(r" {2,}", " ", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    return cleaned.strip()


def _infer_heading(line: str, source_type: str) -> tuple[int, str] | None:
    stripped = line.strip()
    if not stripped:
        return None

    if source_type == "md":
        match = _MARKDOWN_HEADING_RE.match(stripped)
        if match:
            return len(match.group(1)), match.group(2).strip()
        return None

    match = _CHAPTER_HEADING_RE.match(stripped)
    if match:
        title = stripped if not match.group(2) else f"{match.group(1)} {match.group(2).strip()}"
        return 1, title.strip()

    match = _NUMBERED_HEADING_RE.match(stripped)
    if match:
        level = min(match.group(1).count(".") + 1, 4)
        return level, stripped

    match = _YEAR_RANGE_HEADING_RE.match(stripped)
    if match and len(stripped) <= 60:
        return 3, stripped

    if len(stripped) <= 40 and stripped == stripped.title() and stripped[-1] not in ".。:：;；!?！？":
        return 2, stripped

    return None


def _is_markdown_fence_line(line: str) -> bool:
    stripped = line.lstrip()
    return stripped.startswith("```") or stripped.startswith("~~~")


def _split_markdown_logical_blocks(text: str) -> list[str]:
    blocks: list[str] = []
    current_lines: list[str] = []
    code_lines: list[str] = []
    in_code_block = False

    def flush_current() -> None:
        nonlocal current_lines
        block = "\n".join(current_lines).strip()
        if block:
            blocks.append(block)
        current_lines = []

    for raw_line in text.split("\n"):
        if _is_markdown_fence_line(raw_line):
            if in_code_block:
                code_lines.append(raw_line)
                block = "\n".join(code_lines).strip()
                if block:
                    blocks.append(block)
                code_lines = []
                in_code_block = False
            else:
                flush_current()
                code_lines = [raw_line]
                in_code_block = True
            continue

        if in_code_block:
            code_lines.append(raw_line)
            continue

        stripped = raw_line.strip()
        if not stripped:
            flush_current()
            continue

        if _MARKDOWN_HEADING_RE.match(stripped):
            flush_current()
            blocks.append(stripped)
            continue

        current_lines.append(raw_line)

    if code_lines:
        block = "\n".join(code_lines).strip()
        if block:
            blocks.append(block)
    flush_current()
    return blocks


def _build_markdown_chunk_documents(
    doc: Document,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    text = doc.page_content.strip()
    if not text:
        return []

    blocks = _split_markdown_logical_blocks(text)
    if not blocks:
        return []

    chunks: list[Document] = []
    current_blocks: list[str] = []

    def current_text() -> str:
        return "\n\n".join(current_blocks).strip()

    def emit_chunk(content: str) -> None:
        if not content:
            return
        metadata = dict(doc.metadata)
        start_index = text.find(content)
        if start_index >= 0:
            metadata["start_index"] = start_index
        chunks.append(Document(page_content=content, metadata=metadata))

    for block in blocks:
        if block.startswith("```") or block.startswith("~~~"):
            if current_blocks and len(current_text()) + 2 + len(block) > chunk_size:
                emitted = current_text()
                emit_chunk(emitted)
                overlap_blocks: list[str] = []
                overlap_chars = 0
                for tail_block in reversed(current_blocks):
                    if tail_block.startswith("```") or tail_block.startswith("~~~"):
                        continue
                    overlap_blocks.insert(0, tail_block)
                    overlap_chars += len(tail_block)
                    if overlap_chars >= chunk_overlap:
                        break
                current_blocks = overlap_blocks
            current_blocks.append(block)
            if len(block) >= chunk_size:
                emit_chunk(current_text())
                current_blocks = []
            continue

        candidate_blocks = [*current_blocks, block]
        candidate_text = "\n\n".join(candidate_blocks).strip()
        if current_blocks and len(candidate_text) > chunk_size:
            emitted = current_text()
            emit_chunk(emitted)
            overlap_blocks = []
            overlap_chars = 0
            for tail_block in reversed(current_blocks):
                if tail_block.startswith("```") or tail_block.startswith("~~~"):
                    continue
                overlap_blocks.insert(0, tail_block)
                overlap_chars += len(tail_block)
                if overlap_chars >= chunk_overlap:
                    break
            current_blocks = overlap_blocks
            if len(block) > chunk_size:
                emit_chunk(block)
                current_blocks = []
            else:
                current_blocks.append(block)
            continue

        current_blocks = candidate_blocks

    emit_chunk(current_text())
    return chunks


def _apply_structure_metadata(
    metadata: dict[str, object],
    document_structure: str,
    structure_metrics: dict[str, int | float],
) -> None:
    metadata["document_structure"] = document_structure
    metadata["chunk_strategy"] = document_structure
    metadata["structure_heading_lines"] = int(structure_metrics.get("heading_lines", 0))
    metadata["structure_table_lines"] = int(structure_metrics.get("table_lines", 0))
    metadata["structure_key_value_lines"] = int(structure_metrics.get("key_value_lines", 0))
    metadata["structure_paragraphs"] = int(structure_metrics.get("paragraphs", 0))
    metadata["structure_long_paragraphs"] = int(structure_metrics.get("long_paragraphs", 0))
    metadata["structure_nonempty_lines"] = int(structure_metrics.get("nonempty_lines", 0))


def _build_section_documents(docs: list[Document]) -> list[Document]:
    section_docs: list[Document] = []

    for doc_index, doc in enumerate(docs):
        source_type = _detect_source_type(doc.metadata)
        document_structure, structure_metrics = _classify_document_structure(doc.page_content, source_type)
        cleaned_text = _clean_text(doc.page_content, source_type, document_structure)
        if not cleaned_text:
            continue

        logger.info(
            "[indexing.structure] source=%s source_type=%s structure=%s headings=%s table_lines=%s paragraphs=%s",
            doc.metadata.get("source", ""),
            source_type,
            document_structure,
            structure_metrics["heading_lines"],
            structure_metrics["table_lines"],
            structure_metrics["paragraphs"],
        )

        if document_structure == "semi_structured":
            blocks = [block.strip() for block in re.split(r"\n{2,}", cleaned_text) if block.strip()]
            for block_index, block in enumerate(blocks or [cleaned_text], start=1):
                metadata = dict(doc.metadata)
                metadata["source_type"] = source_type
                _apply_structure_metadata(metadata, document_structure, structure_metrics)
                metadata["layout_parser"] = "table-aware-layout"
                metadata["section_path"] = (
                    f"table_block_{block_index}" if len(blocks) > 1 else (doc.metadata.get("section_path") or "root")
                )
                metadata["document_index"] = doc_index
                metadata["cleaned_chars"] = len(block)
                section_docs.append(Document(page_content=block, metadata=metadata))
            continue

        heading_stack: list[str] = []
        current_buffer: list[str] = []
        current_section_path = doc.metadata.get("section_path") or "root"
        in_code_block = False

        def flush_section() -> None:
            nonlocal current_buffer
            section_text = "\n".join(current_buffer).strip()
            if not section_text:
                current_buffer = []
                return

            metadata = dict(doc.metadata)
            metadata["source_type"] = source_type
            _apply_structure_metadata(metadata, document_structure, structure_metrics)
            metadata["layout_parser"] = "markdown-headings" if source_type == "md" else "heuristic-layout"
            metadata["section_path"] = current_section_path
            metadata["document_index"] = doc_index
            metadata["cleaned_chars"] = len(section_text)
            section_docs.append(Document(page_content=section_text, metadata=metadata))
            current_buffer = []

        for raw_line in cleaned_text.split("\n"):
            if source_type == "md" and _is_markdown_fence_line(raw_line):
                current_buffer.append(raw_line)
                in_code_block = not in_code_block
                continue

            if in_code_block:
                current_buffer.append(raw_line)
                continue

            stripped = raw_line.strip()
            heading = _infer_heading(stripped, source_type)
            if heading:
                flush_section()
                level, title = heading
                while len(heading_stack) >= level:
                    heading_stack.pop()
                heading_stack.append(title)
                current_section_path = "/".join(heading_stack) if heading_stack else "root"
                continue

            current_buffer.append(raw_line)

        flush_section()

        if not section_docs or section_docs[-1].metadata.get("document_index") != doc_index:
            metadata = dict(doc.metadata)
            metadata["source_type"] = source_type
            _apply_structure_metadata(metadata, document_structure, structure_metrics)
            metadata["layout_parser"] = "fallback-whole-document"
            metadata["section_path"] = current_section_path
            metadata["document_index"] = doc_index
            metadata["cleaned_chars"] = len(cleaned_text)
            section_docs.append(Document(page_content=cleaned_text, metadata=metadata))

    return section_docs


def _resolve_chunk_profile_key(metadata: dict[str, object]) -> str:
    document_structure = str(metadata.get("document_structure", "")).strip()
    if document_structure in _STRUCTURE_CHUNKING_PROFILES:
        return document_structure
    source_type = _detect_source_type(metadata)
    if source_type in _CHUNKING_PROFILES:
        return source_type
    return "default"


def _get_chunking_profile(profile_key: str) -> dict[str, object]:
    if profile_key in _STRUCTURE_CHUNKING_PROFILES:
        return _STRUCTURE_CHUNKING_PROFILES[profile_key]
    return _CHUNKING_PROFILES.get(profile_key, _CHUNKING_PROFILES["default"])


def _build_splitter(profile_key: str, level: str) -> RecursiveCharacterTextSplitter:
    profile = _get_chunking_profile(profile_key)
    separators = profile["separators"]
    if level == "parent":
        chunk_size = int(profile["parent_size"])
        chunk_overlap = int(profile["parent_overlap"])
    else:
        chunk_size = int(profile["child_size"])
        chunk_overlap = int(profile["child_overlap"])
    return RecursiveCharacterTextSplitter(
        separators=separators,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )


def _get_chunk_profile(profile_key: str, level: str) -> tuple[int, int]:
    profile = _get_chunking_profile(profile_key)
    if level == "parent":
        return int(profile["parent_size"]), int(profile["parent_overlap"])
    return int(profile["child_size"]), int(profile["child_overlap"])


def _stable_chunk_id(metadata: dict[str, object], content: str, level: str, index: int) -> str:
    digest_input = "|".join(
        [
            str(metadata.get("source", "")),
            str(metadata.get("section_path", "root")),
            str(metadata.get("page", "")),
            level,
            str(index),
            content,
        ]
    )
    return hashlib.sha1(digest_input.encode("utf-8")).hexdigest()[:20]


def _inject_section_context(content: str, metadata: dict[str, object]) -> tuple[str, str]:
    section_path = str(metadata.get("section_path", "root")).strip() or "root"
    if section_path == "root":
        return content, content

    section_title = section_path.split("/")[-1]
    source_type = _detect_source_type(metadata)
    if source_type == "md":
        prefix = f"Section path: {section_path}\nSection title: {section_title}\n\n"
    else:
        prefix = f"[Section Path] {section_path}\n[Section Title] {section_title}\n\n"

    return prefix + content, content


def _build_document_id(metadata: dict[str, object]) -> str:
    digest_input = "|".join(
        [
            str(metadata.get("source", "")),
            str(metadata.get("source_type", "")),
            str(metadata.get("page", "")),
        ]
    )
    return hashlib.sha1(digest_input.encode("utf-8")).hexdigest()[:16]


def _extract_title_and_tags(metadata: dict[str, object], content: str) -> tuple[str, list[str]]:
    section_path = str(metadata.get("section_path", "root")).strip() or "root"
    source = str(metadata.get("source", "")).strip()
    source_stem = Path(source).stem.strip()
    title = section_path.split("/")[-1].strip() if section_path != "root" else source_stem or "root"

    tag_candidates: list[str] = []
    if source_stem:
        tag_candidates.extend(token for token in _TAG_SPLIT_RE.split(source_stem) if token)
    if section_path != "root":
        tag_candidates.extend(part.strip() for part in section_path.split("/") if part.strip())

    first_line = next((line.strip() for line in content.splitlines() if line.strip()), "")
    if first_line and len(first_line) <= 80:
        tag_candidates.extend(token for token in _TAG_SPLIT_RE.split(first_line) if token)

    tag_candidates.extend(
        [
            str(metadata.get("source_type", "")).strip(),
            str(metadata.get("chunk_level", "")).strip(),
        ]
    )

    seen: set[str] = set()
    tags: list[str] = []
    for candidate in tag_candidates:
        normalized = candidate.strip().lower()
        if len(normalized) < 2 or len(normalized) > 32:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        tags.append(normalized)
        if len(tags) >= 8:
            break
    return title or "root", tags


def _build_parent_child_chunks(section_docs: list[Document]) -> tuple[list[Document], list[Document]]:
    parent_chunks: list[Document] = []
    child_chunks: list[Document] = []

    for section_index, section_doc in enumerate(section_docs):
        source_type = _detect_source_type(section_doc.metadata)
        profile_key = _resolve_chunk_profile_key(section_doc.metadata)
        parent_chunk_size, parent_chunk_overlap = _get_chunk_profile(profile_key, "parent")
        child_chunk_size, child_chunk_overlap = _get_chunk_profile(profile_key, "child")
        if source_type == "md":
            parent_docs = _build_markdown_chunk_documents(section_doc, parent_chunk_size, parent_chunk_overlap)
        else:
            parent_splitter = _build_splitter(profile_key, "parent")
            parent_docs = parent_splitter.split_documents([section_doc])

        for parent_index, parent_doc in enumerate(parent_docs):
            parent_metadata = dict(parent_doc.metadata)
            parent_id = _stable_chunk_id(parent_metadata, parent_doc.page_content, "parent", parent_index)
            injected_parent_content, parent_preview = _inject_section_context(
                parent_doc.page_content,
                parent_metadata,
            )
            child_seed = Document(page_content=parent_preview, metadata=parent_metadata)
            if source_type == "md":
                child_docs = _build_markdown_chunk_documents(child_seed, child_chunk_size, child_chunk_overlap)
            else:
                child_splitter = _build_splitter(profile_key, "child")
                child_docs = child_splitter.split_documents([child_seed])
            child_chunk_count = len(child_docs)
            parent_metadata["chunk_level"] = "parent"
            parent_title, parent_tags = _extract_title_and_tags(parent_metadata, parent_preview)
            parent_metadata.update(
                {
                    "document_id": _build_document_id(parent_metadata),
                    "title": parent_title,
                    "tags": parent_tags,
                    "parent_chunk_id": parent_id,
                    "parent_chunk_index": parent_index,
                    "section_index": section_index,
                    "section_depth": len(str(parent_metadata.get("section_path", "root")).split("/")),
                    "parent_section_path": parent_metadata.get("section_path", "root"),
                    "chunk_strategy": profile_key,
                    "chunk_size": parent_chunk_size,
                    "chunk_overlap": parent_chunk_overlap,
                    "has_children": child_chunk_count > 0,
                    "child_chunk_count": child_chunk_count,
                    "child_chunk_start_index": 0 if child_chunk_count else None,
                    "child_chunk_end_index": child_chunk_count - 1 if child_chunk_count else None,
                    "content_preview": parent_preview,
                }
            )
            parent_chunks.append(Document(page_content=injected_parent_content, metadata=parent_metadata))

            for child_index, child_doc in enumerate(child_docs):
                child_metadata = dict(child_doc.metadata)
                injected_child_content, child_preview = _inject_section_context(
                    child_doc.page_content,
                    child_metadata,
                )
                child_metadata["chunk_level"] = "child"
                child_title, child_tags = _extract_title_and_tags(child_metadata, child_preview)
                child_metadata.update(
                    {
                        "document_id": _build_document_id(child_metadata),
                        "title": child_title,
                        "tags": child_tags,
                        "parent_chunk_id": parent_id,
                        "child_chunk_index": child_index,
                        "section_path": child_metadata.get("section_path", "root"),
                        "parent_section_path": child_metadata.get("parent_section_path", "root"),
                        "section_depth": len(str(child_metadata.get("section_path", "root")).split("/")),
                        "chunk_strategy": profile_key,
                        "chunk_size": child_chunk_size,
                        "chunk_overlap": child_chunk_overlap,
                        "has_children": False,
                        "parent_content_preview": parent_preview[:200],
                        "content_preview": child_preview,
                        "chunk_id": _stable_chunk_id(child_metadata, injected_child_content, "child", child_index),
                    }
                )
                child_chunks.append(Document(page_content=injected_child_content, metadata=child_metadata))

    return parent_chunks, child_chunks


def _build_loader(loader_cls: type, file_path: str):
    if loader_cls is TextLoader:
        return loader_cls(file_path, encoding="utf-8", autodetect_encoding=True)
    return loader_cls(file_path)


def _get_embeddings():
    if "dashscope.aliyuncs.com" in settings.embedding_base_url:
        logger.info("[indexing.embeddings] provider=dashscope model=%s", settings.embedding_model)
        return DashScopeEmbeddings(
            model=settings.embedding_model,
            dashscope_api_key=settings.embedding_api_key,
        )

    logger.info(
        "[indexing.embeddings] provider=openai-compatible model=%s base_url=%s",
        settings.embedding_model,
        settings.embedding_base_url,
    )
    kwargs = {
        "model": settings.embedding_model,
        "api_key": settings.embedding_api_key,
        "base_url": settings.embedding_base_url,
    }
    if settings.embedding_dimensions is not None:
        kwargs["dimensions"] = settings.embedding_dimensions
    return OpenAIEmbeddings(**kwargs)


def _get_chroma_persist_dir() -> str:
    """Return a stable absolute path for the Chroma persistence directory."""
    return str(Path(settings.chroma_persist_dir).expanduser().resolve())


def get_vectorstore(collection_name: str | None = None) -> Chroma:
    """Return (or create) a Chroma vector store for the given collection."""
    name = collection_name or settings.chroma_collection_name
    persist_dir = _get_chroma_persist_dir()
    logger.info("[indexing.vectorstore] open collection=%s persist_dir=%s", name, persist_dir)
    return Chroma(
        collection_name=name,
        embedding_function=_get_embeddings(),
        persist_directory=persist_dir,
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
    logger.info(
        "[indexing.file] start filename=%s collection=%s bytes=%s",
        filename,
        collection_name or settings.chroma_collection_name,
        len(file_content),
    )
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
        loader = _build_loader(loader_cls, tmp_path)
        docs: list[Document] = loader.load()
    finally:
        os.unlink(tmp_path)

    logger.info("[indexing.file] loaded filename=%s docs=%s", filename, len(docs))

    # Always expose the original uploaded filename instead of the temp loader path.
    for doc in docs:
        doc.metadata["source"] = filename
        doc.metadata["source_type"] = suffix.lstrip(".") or "text"

    return _split_and_store(docs, collection_name)


def index_text(
    text: str,
    source: str = "inline",
    collection_name: str | None = None,
) -> int:
    """Index a raw text string and return the chunk count."""
    logger.info(
        "[indexing.text] start source=%s collection=%s text_chars=%s",
        source,
        collection_name or settings.chroma_collection_name,
        len(text),
    )
    source_type = Path(source).suffix.lower().lstrip(".") or "text"
    docs = [Document(page_content=text, metadata={"source": source, "source_type": source_type})]
    return _split_and_store(docs, collection_name)


def _split_and_store(
    docs: list[Document],
    collection_name: str | None = None,
) -> int:
    """Split *docs* into chunks and persist them.  Returns chunk count."""
    logger.info(
        "[indexing.split] start docs=%s collection=%s",
        len(docs),
        collection_name or settings.chroma_collection_name,
    )
    section_docs = _build_section_documents(docs)
    logger.info("[indexing.layout] section_docs=%s", len(section_docs))
    parent_chunks, child_chunks = _build_parent_child_chunks(section_docs)
    logger.info(
        "[indexing.split] produced parents=%s children=%s",
        len(parent_chunks),
        len(child_chunks),
    )
    all_chunks = [*parent_chunks, *child_chunks]
    if not all_chunks:
        logger.warning("[indexing.split] no chunks produced")
        return 0

    vs = get_vectorstore(collection_name)
    logger.info(
        "[indexing.store] add_documents parent_chunks=%s child_chunks=%s total=%s",
        len(parent_chunks),
        len(child_chunks),
        len(all_chunks),
    )
    vs.add_documents(all_chunks)
    logger.info("[indexing.store] complete total_chunks=%s", len(all_chunks))
    return len(all_chunks)


def list_collections() -> list[dict[str, int | str]]:
    """Return all Chroma collections with their document counts."""
    import chromadb

    client = chromadb.PersistentClient(path=_get_chroma_persist_dir())
    result = []
    for col in client.list_collections():
        result.append({"name": col.name, "count": col.count()})
    return result


def list_chunks(
    collection_name: str | None = None,
    limit: int = 20,
    offset: int = 0,
    source_filter: str | None = None,
) -> dict[str, object]:
    """Return paginated stored chunks from a Chroma collection."""
    vs = get_vectorstore(collection_name)
    resolved_collection = collection_name or settings.chroma_collection_name
    where_filter = {"source": source_filter} if source_filter else None
    logger.info(
        "[indexing.chunks] list collection=%s limit=%s offset=%s source=%s",
        resolved_collection,
        limit,
        offset,
        source_filter or "*",
    )
    collection = vs._collection
    total_payload = collection.get(where=where_filter, include=[])
    total = len(total_payload.get("ids") or [])
    payload = collection.get(
        where=where_filter,
        limit=limit,
        offset=offset,
        include=["documents", "metadatas"],
    )
    ids = payload.get("ids") or []
    documents = payload.get("documents") or []
    metadatas = payload.get("metadatas") or []
    chunks: list[dict[str, object]] = []
    for chunk_id, document, metadata in zip(ids, documents, metadatas):
        metadata = metadata or {}
        chunks.append(
            {
                "id": chunk_id,
                "source": metadata.get("source", ""),
                "page": metadata.get("page"),
                "source_type": metadata.get("source_type"),
                "chunk_level": metadata.get("chunk_level"),
                "title": metadata.get("title"),
                "tags": metadata.get("tags") or [],
                "section_path": metadata.get("section_path"),
                "parent_section_path": metadata.get("parent_section_path"),
                "parent_chunk_id": metadata.get("parent_chunk_id"),
                "parent_chunk_index": metadata.get("parent_chunk_index"),
                "child_chunk_index": metadata.get("child_chunk_index"),
                "child_chunk_count": metadata.get("child_chunk_count"),
                "child_chunk_start_index": metadata.get("child_chunk_start_index"),
                "child_chunk_end_index": metadata.get("child_chunk_end_index"),
                "document_id": metadata.get("document_id"),
                "section_depth": metadata.get("section_depth"),
                "chunk_size": metadata.get("chunk_size"),
                "chunk_overlap": metadata.get("chunk_overlap"),
                "has_children": metadata.get("has_children"),
                "start_index": metadata.get("start_index"),
                "content": document or "",
            }
        )
    return {
        "collection_name": resolved_collection,
        "source_filter": source_filter,
        "total": total,
        "limit": limit,
        "offset": offset,
        "chunks": chunks,
    }


def get_parent_chunks_by_ids(
    parent_chunk_ids: list[str],
    collection_name: str | None = None,
) -> list[Document]:
    """Return parent chunks keyed by their metadata parent_chunk_id."""
    normalized_ids = [item.strip() for item in parent_chunk_ids if str(item).strip()]
    if not normalized_ids:
        return []

    vs = get_vectorstore(collection_name)
    resolved_collection = collection_name or settings.chroma_collection_name
    logger.info(
        "[indexing.parents] fetch collection=%s parent_ids=%s",
        resolved_collection,
        len(normalized_ids),
    )
    collection = vs._collection
    payload = collection.get(
        where={
            "$and": [
                {"chunk_level": "parent"},
                {"parent_chunk_id": {"$in": normalized_ids}},
            ]
        },
        include=["documents", "metadatas"],
    )
    documents = payload.get("documents") or []
    metadatas = payload.get("metadatas") or []
    parent_docs: list[Document] = []
    for document, metadata in zip(documents, metadatas):
        parent_docs.append(Document(page_content=document or "", metadata=metadata or {}))
    return parent_docs


def delete_document(doc_id: str, collection_name: str | None = None) -> None:
    """Delete a document by its Chroma *doc_id*."""
    vs = get_vectorstore(collection_name)
    logger.info(
        "[indexing.delete] delete doc_id=%s collection=%s",
        doc_id,
        collection_name or settings.chroma_collection_name,
    )
    vs.delete([doc_id])


def delete_chunks_by_source(source: str, collection_name: str | None = None) -> int:
    """Delete all chunks whose metadata source matches the provided value."""
    normalized_source = source.strip()
    if not normalized_source:
        raise ValueError("Source is required.")

    vs = get_vectorstore(collection_name)
    resolved_collection = collection_name or settings.chroma_collection_name
    collection = vs._collection
    logger.info(
        "[indexing.delete] delete source=%s collection=%s",
        normalized_source,
        resolved_collection,
    )
    payload = collection.get(where={"source": normalized_source}, include=[])
    ids = payload.get("ids") or []
    if not ids:
        raise KeyError(normalized_source)

    collection.delete(where={"source": normalized_source})
    return len(ids)
