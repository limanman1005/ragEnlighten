"""LangSmith tracing configuration and RunnableConfig helpers."""

from __future__ import annotations

import logging
import os
from typing import Any

from app.core import config

logger = logging.getLogger("uvicorn.error")

_CONFIGURED = False

REACT_AGENT_QUERY_ENDPOINT = "/api/v1/chat/react-agent"
REACT_AGENT_STREAM_ENDPOINT = "/api/v1/chat/react-agent/stream"
PLAN_EXECUTE_QUERY_ENDPOINT = "/api/v1/chat/plan-execute-agent"
PLAN_EXECUTE_STREAM_ENDPOINT = "/api/v1/chat/plan-execute-agent/stream"


def configure_langsmith_tracing() -> None:
    """Apply LangSmith/LangChain tracing environment variables at process startup."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    _CONFIGURED = True

    if not config.settings.langsmith_tracing_enabled:
        logger.info("[tracing] LangSmith tracing is disabled")
        return

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    if config.settings.langsmith_api_key:
        os.environ["LANGCHAIN_API_KEY"] = config.settings.langsmith_api_key
    if config.settings.langsmith_project:
        os.environ["LANGCHAIN_PROJECT"] = config.settings.langsmith_project
    if config.settings.langsmith_endpoint:
        os.environ["LANGCHAIN_ENDPOINT"] = config.settings.langsmith_endpoint

    logger.info(
        "[tracing] LangSmith tracing enabled project=%s",
        config.settings.langsmith_project,
    )


def is_tracing_enabled() -> bool:
    return config.settings.langsmith_tracing_enabled


def build_agent_metadata(
    *,
    route: str,
    endpoint: str,
    streaming: bool,
    collection_name: str | None = None,
    history_count: int | None = None,
    phase: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build unified LangSmith metadata for agent runs."""
    metadata: dict[str, Any] = {
        "route": route,
        "endpoint": endpoint,
        "streaming": streaming,
        "collection_name": collection_name or config.settings.chroma_collection_name,
    }
    if history_count is not None:
        metadata["history_count"] = history_count
    if phase:
        metadata["phase"] = phase
    if extra:
        metadata.update(extra)
    return metadata


def build_run_config(
    run_name: str,
    *,
    recursion_limit: int | None = None,
    route: str | None = None,
    endpoint: str | None = None,
    streaming: bool | None = None,
    collection_name: str | None = None,
    history_count: int | None = None,
    phase: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a LangChain/LangGraph config dict with optional LangSmith run metadata."""
    config: dict[str, Any] = {}
    if recursion_limit is not None:
        config["recursion_limit"] = recursion_limit

    if not is_tracing_enabled():
        return config

    run_metadata = build_agent_metadata(
        route=route or "unknown",
        endpoint=endpoint or "unknown",
        streaming=bool(streaming),
        collection_name=collection_name,
        history_count=history_count,
        phase=phase,
        extra=metadata,
    )

    config["run_name"] = run_name
    config["tags"] = [run_name]
    config["metadata"] = run_metadata
    return config


def merge_run_configs(*configs: dict[str, Any]) -> dict[str, Any]:
    """Merge multiple run config dicts (metadata and tags are combined)."""
    merged: dict[str, Any] = {}
    for cfg in configs:
        if not cfg:
            continue
        for key, value in cfg.items():
            if key == "metadata" and isinstance(value, dict):
                existing = merged.get("metadata")
                if isinstance(existing, dict):
                    merged["metadata"] = {**existing, **value}
                else:
                    merged["metadata"] = dict(value)
            elif key == "tags" and isinstance(value, list):
                merged["tags"] = [*merged.get("tags", []), *value]
            else:
                merged[key] = value
    return merged
