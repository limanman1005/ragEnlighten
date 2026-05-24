"""Server-Sent Events formatting helpers."""

from __future__ import annotations

import json
from typing import Any


def format_sse_event(event_type: str, data: Any) -> str:
    """Format one SSE event frame."""
    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
