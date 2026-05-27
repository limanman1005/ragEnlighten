"""LangSmith tracing configuration and RunnableConfig helpers."""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Generator

from app.core import config

logger = logging.getLogger("uvicorn.error")

_CONFIGURED = False

REACT_AGENT_QUERY_ENDPOINT = "/api/v1/chat/react-agent"
REACT_AGENT_STREAM_ENDPOINT = "/api/v1/chat/react-agent/stream"
PLAN_EXECUTE_QUERY_ENDPOINT = "/api/v1/chat/plan-execute-agent"
PLAN_EXECUTE_STREAM_ENDPOINT = "/api/v1/chat/plan-execute-agent/stream"

PLAN_EXECUTE_EXECUTABLE_TOOLS = frozenset(
    {"knowledge_base_search", "collection_overview", "web_search"}
)


@dataclass
class AgentTraceContext:
    """Request-scoped LangSmith metadata and optional parent run reference."""

    route: str
    endpoint: str
    streaming: bool
    collection_name: str
    history_count: int = 0
    parent_run: Any | None = field(default=None, repr=False)

    def metadata_for_phase(
        self,
        phase: str | None = None,
        *,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return build_agent_metadata(
            route=self.route,
            endpoint=self.endpoint,
            streaming=self.streaming,
            collection_name=self.collection_name,
            history_count=self.history_count,
            phase=phase,
            extra=extra,
        )


def configure_langsmith_tracing() -> None:
    """Apply LangSmith/LangChain tracing environment variables at process startup."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    _CONFIGURED = True

    if not config.settings.langsmith_tracing_enabled:
        logger.info("[tracing] LangSmith tracing is disabled")
        return

    if not config.settings.langsmith_api_key.strip():
        logger.warning(
            "[tracing] LANGSMITH_TRACING_ENABLED is true but LANGSMITH_API_KEY is empty; "
            "LangSmith export will remain off"
        )
        return

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
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
    """Return True only when tracing is requested and an API key is configured."""
    return bool(
        config.settings.langsmith_tracing_enabled
        and config.settings.langsmith_api_key.strip()
    )


def build_agent_trace_context(
    *,
    route: str,
    endpoint: str,
    streaming: bool,
    collection_name: str | None = None,
    history_count: int = 0,
) -> AgentTraceContext:
    return AgentTraceContext(
        route=route,
        endpoint=endpoint,
        streaming=streaming,
        collection_name=collection_name or config.settings.chroma_collection_name,
        history_count=history_count,
    )


@contextmanager
def agent_request_tracing(
    run_name: str,
    trace_context: AgentTraceContext,
) -> Generator[AgentTraceContext, None, None]:
    """Open a root LangSmith run and set parent context for nested child runs."""
    if not is_tracing_enabled():
        yield trace_context
        return

    from langsmith.run_helpers import trace, tracing_context

    metadata = trace_context.metadata_for_phase()
    with trace(name=run_name, run_type="chain", metadata=metadata) as root_run:
        trace_context.parent_run = root_run
        with tracing_context(parent=root_run, metadata=metadata):
            yield trace_context


@contextmanager
def agent_child_trace(
    name: str,
    trace_context: AgentTraceContext,
    *,
    run_type: str = "chain",
    phase: str | None = None,
    inputs: dict[str, Any] | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> Generator[Any, None, None]:
    """Create a nested LangSmith run with unified request metadata."""
    if not is_tracing_enabled():
        yield None
        return

    from langsmith.run_helpers import trace

    metadata = trace_context.metadata_for_phase(phase, extra=extra_metadata)
    parent = trace_context.parent_run
    with trace(
        name=name,
        run_type=run_type,
        inputs=inputs or {},
        metadata=metadata,
        parent=parent,
    ) as child_run:
        yield child_run


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
    trace_context: AgentTraceContext | None = None,
) -> dict[str, Any]:
    """Build a LangChain/LangGraph config dict with optional LangSmith run metadata."""
    run_config: dict[str, Any] = {}
    if recursion_limit is not None:
        run_config["recursion_limit"] = recursion_limit

    if not is_tracing_enabled():
        return run_config

    if trace_context is not None:
        run_metadata = trace_context.metadata_for_phase(phase, extra=metadata)
    else:
        run_metadata = build_agent_metadata(
            route=route or "unknown",
            endpoint=endpoint or "unknown",
            streaming=bool(streaming),
            collection_name=collection_name,
            history_count=history_count,
            phase=phase,
            extra=metadata,
        )

    run_config["run_name"] = run_name
    run_config["tags"] = [run_name]
    run_config["metadata"] = run_metadata
    return run_config


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
