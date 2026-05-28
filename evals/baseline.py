"""Baseline offline evaluation for React and Plan-Execute agents."""

from __future__ import annotations

import csv
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable


REACT_MODE = "react"
PLAN_EXECUTE_MODE = "plan_execute"
SUPPORTED_MODES = (REACT_MODE, PLAN_EXECUTE_MODE)

MODE_TO_ENDPOINT = {
    REACT_MODE: "/api/v1/chat/react-agent",
    PLAN_EXECUTE_MODE: "/api/v1/chat/plan-execute-agent",
}
MODE_TO_ROUTE = {
    REACT_MODE: "react_agent",
    PLAN_EXECUTE_MODE: "plan_execute_agent",
}

_REFUSAL_MARKERS = (
    "could not verify",
    "i could not find grounded evidence",
    "not returning a factual answer",
    "please refine the query",
    "无法",
    "未找到",
    "不返回事实性答案",
    "请调整问题",
)

_TRACE_ID_KEYS = (
    "trace_id",
    "run_id",
    "langsmith_run_id",
    "smith_run_id",
)


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    question: str
    collection_name: str | None = None
    history: list[dict[str, Any]] | None = None
    expected_tools: list[str] | None = None
    should_be_grounded: bool | None = None
    should_refuse_when_no_evidence: bool = False


@dataclass
class EvalOutcome:
    case: EvalCase
    mode: str
    endpoint: str
    success: bool
    status_code: int
    duration_ms: int
    response: dict[str, Any] | None = None
    error: str | None = None


@dataclass
class ScoredOutcome:
    case_id: str
    mode: str
    endpoint: str
    request_success: bool
    status_code: int
    duration_ms: int
    tool_expectation_applicable: bool
    tool_expectation_hit: bool | None
    grounding_applicable: bool
    grounding_pass: bool | None
    no_evidence_applicable: bool
    no_evidence_answer_leakage: bool | None
    expected_tools: list[str]
    observed_tools: list[str]
    route: str
    streaming: bool
    collection_name: str
    phase: str | None
    trace: list[str]
    trace_ref: str | None
    answer_preview: str
    validation_passed: bool
    citations_verified: bool
    sources_count: int
    error: str | None


@dataclass
class ModeSummary:
    mode: str
    total_cases: int
    request_success_rate: float
    tool_expectation_hit_rate: float | None
    grounding_pass_rate: float | None
    no_evidence_answer_rate: float | None
    request_success_count: int
    tool_expectation_denominator: int
    grounding_denominator: int
    no_evidence_denominator: int


def load_cases(dataset_path: str | Path) -> list[EvalCase]:
    path = Path(dataset_path)
    cases: list[EvalCase] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        payload = json.loads(line)
        case_id = str(payload.get("case_id") or "").strip()
        question = str(payload.get("question") or "").strip()
        if not case_id:
            raise ValueError(f"Line {line_number}: case_id is required")
        if not question:
            raise ValueError(f"Line {line_number}: question is required")
        expected_tools = payload.get("expected_tools")
        if expected_tools is not None and not isinstance(expected_tools, list):
            raise ValueError(f"Line {line_number}: expected_tools must be a list")
        history = payload.get("history")
        if history is not None and not isinstance(history, list):
            raise ValueError(f"Line {line_number}: history must be a list")
        cases.append(
            EvalCase(
                case_id=case_id,
                question=question,
                collection_name=payload.get("collection_name"),
                history=history,
                expected_tools=[str(item) for item in expected_tools] if expected_tools else [],
                should_be_grounded=payload.get("should_be_grounded"),
                should_refuse_when_no_evidence=bool(payload.get("should_refuse_when_no_evidence", False)),
            )
        )
    if not cases:
        raise ValueError(f"No cases found in dataset: {path}")
    return cases


def run_cases_http(
    cases: Iterable[EvalCase],
    *,
    base_url: str,
    timeout_seconds: float,
    modes: Iterable[str],
) -> list[EvalOutcome]:
    outcomes: list[EvalOutcome] = []
    normalized_base_url = base_url.rstrip("/")
    for case in cases:
        for mode in modes:
            if mode not in SUPPORTED_MODES:
                raise ValueError(f"Unsupported mode: {mode}")
            endpoint = MODE_TO_ENDPOINT[mode]
            outcomes.append(
                _call_endpoint(
                    case,
                    mode=mode,
                    endpoint=endpoint,
                    url=f"{normalized_base_url}{endpoint}",
                    timeout_seconds=timeout_seconds,
                )
            )
    return outcomes


def score_outcomes(outcomes: Iterable[EvalOutcome]) -> list[ScoredOutcome]:
    scored: list[ScoredOutcome] = []
    for outcome in outcomes:
        response = outcome.response or {}
        expected_tools = [tool.lower() for tool in (outcome.case.expected_tools or [])]
        observed_tools = sorted(set(_extract_tool_names(response)))

        tool_expectation_applicable = bool(expected_tools)
        tool_expectation_hit = None
        if tool_expectation_applicable:
            expected = set(expected_tools)
            tool_expectation_hit = expected.issubset(set(observed_tools))

        grounding_applicable = outcome.case.should_be_grounded is not None
        grounding_pass = None
        if grounding_applicable:
            grounding_pass = _is_grounded(response) == bool(outcome.case.should_be_grounded)

        no_evidence_applicable = outcome.case.should_refuse_when_no_evidence
        no_evidence_answer_leakage = None
        if no_evidence_applicable:
            no_evidence_answer_leakage = _is_no_evidence_answer_leakage(response)

        phase = _extract_primary_phase(response)
        trace_ref = _extract_trace_reference(response)
        route = str(response.get("route") or MODE_TO_ROUTE[outcome.mode])
        collection_name = str(response.get("collection_name") or outcome.case.collection_name or "")
        validation = response.get("validation") or {}
        citations_verified = bool(validation.get("citations_verified", False))
        validation_passed = bool(validation.get("passed", False))
        sources = response.get("sources") or []
        answer = str(response.get("answer") or "")
        scored.append(
            ScoredOutcome(
                case_id=outcome.case.case_id,
                mode=outcome.mode,
                endpoint=outcome.endpoint,
                request_success=outcome.success,
                status_code=outcome.status_code,
                duration_ms=outcome.duration_ms,
                tool_expectation_applicable=tool_expectation_applicable,
                tool_expectation_hit=tool_expectation_hit,
                grounding_applicable=grounding_applicable,
                grounding_pass=grounding_pass,
                no_evidence_applicable=no_evidence_applicable,
                no_evidence_answer_leakage=no_evidence_answer_leakage,
                expected_tools=expected_tools,
                observed_tools=observed_tools,
                route=route,
                streaming=False,
                collection_name=collection_name,
                phase=phase,
                trace=[str(item) for item in (response.get("trace") or [])],
                trace_ref=trace_ref,
                answer_preview=answer[:240],
                validation_passed=validation_passed,
                citations_verified=citations_verified,
                sources_count=len(sources) if isinstance(sources, list) else 0,
                error=outcome.error,
            )
        )
    return scored


def build_summary(scored_outcomes: Iterable[ScoredOutcome]) -> dict[str, Any]:
    by_mode: dict[str, list[ScoredOutcome]] = {mode: [] for mode in SUPPORTED_MODES}
    for item in scored_outcomes:
        by_mode.setdefault(item.mode, []).append(item)

    summaries = [asdict(_build_mode_summary(mode, items)) for mode, items in by_mode.items() if items]
    return {"modes": summaries}


def write_results(
    *,
    output_dir: str | Path,
    scored_outcomes: list[ScoredOutcome],
    summary: dict[str, Any],
    export_csv: bool = False,
) -> dict[str, str]:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    results_json_path = path / "results.json"
    summary_json_path = path / "summary.json"
    results_json_path.write_text(
        json.dumps([asdict(item) for item in scored_outcomes], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary_json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    output_paths = {
        "results_json": str(results_json_path),
        "summary_json": str(summary_json_path),
    }
    if export_csv:
        csv_path = path / "results.csv"
        _write_results_csv(csv_path, scored_outcomes)
        output_paths["results_csv"] = str(csv_path)
    return output_paths


def _call_endpoint(
    case: EvalCase,
    *,
    mode: str,
    endpoint: str,
    url: str,
    timeout_seconds: float,
) -> EvalOutcome:
    payload = {"question": case.question, "history": case.history or []}
    if case.collection_name:
        payload["collection_name"] = case.collection_name
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url=url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            raw = response.read().decode("utf-8")
            status_code = int(response.status)
        duration_ms = int((time.monotonic() - started) * 1000)
        parsed = json.loads(raw) if raw else {}
        return EvalOutcome(
            case=case,
            mode=mode,
            endpoint=endpoint,
            success=True,
            status_code=status_code,
            duration_ms=duration_ms,
            response=parsed if isinstance(parsed, dict) else {},
        )
    except urllib.error.HTTPError as exc:
        duration_ms = int((time.monotonic() - started) * 1000)
        error_text = exc.read().decode("utf-8", errors="ignore")
        return EvalOutcome(
            case=case,
            mode=mode,
            endpoint=endpoint,
            success=False,
            status_code=exc.code,
            duration_ms=duration_ms,
            response=None,
            error=error_text or str(exc),
        )
    except Exception as exc:  # noqa: BLE001
        duration_ms = int((time.monotonic() - started) * 1000)
        return EvalOutcome(
            case=case,
            mode=mode,
            endpoint=endpoint,
            success=False,
            status_code=0,
            duration_ms=duration_ms,
            response=None,
            error=str(exc),
        )


def _extract_tool_names(response: dict[str, Any]) -> list[str]:
    tool_calls = response.get("tool_calls") or []
    names: list[str] = []
    if isinstance(tool_calls, list):
        for item in tool_calls:
            if isinstance(item, dict):
                name = str(item.get("name") or "").strip().lower()
                if name:
                    names.append(name)
    return names


def _is_grounded(response: dict[str, Any]) -> bool:
    validation = response.get("validation") or {}
    if bool(validation.get("passed")) and bool(validation.get("citations_verified")):
        return True
    sources = response.get("sources") or []
    return isinstance(sources, list) and len(sources) > 0


def _is_no_evidence_answer_leakage(response: dict[str, Any]) -> bool:
    if _is_grounded(response):
        return False
    answer = str(response.get("answer") or "").strip().lower()
    if not answer:
        return False
    return not any(marker in answer for marker in _REFUSAL_MARKERS)


def _extract_primary_phase(response: dict[str, Any]) -> str | None:
    debug_events = response.get("debug_events") or []
    if not isinstance(debug_events, list):
        return None
    for event in debug_events:
        if isinstance(event, dict) and event.get("phase"):
            return str(event["phase"])
    return None


def _extract_trace_reference(response: dict[str, Any]) -> str | None:
    debug_events = response.get("debug_events") or []
    if isinstance(debug_events, list):
        for event in debug_events:
            if not isinstance(event, dict):
                continue
            details = event.get("details")
            if not isinstance(details, dict):
                continue
            for key in _TRACE_ID_KEYS:
                value = details.get(key)
                if value:
                    return str(value)
    if os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true":
        return "enabled-without-explicit-id"
    return None


def _build_mode_summary(mode: str, items: list[ScoredOutcome]) -> ModeSummary:
    total = len(items)
    request_success_count = sum(1 for item in items if item.request_success)

    tool_items = [item for item in items if item.tool_expectation_applicable]
    grounding_items = [item for item in items if item.grounding_applicable]
    no_evidence_items = [item for item in items if item.no_evidence_applicable]

    def _ratio(num: int, den: int) -> float | None:
        if den == 0:
            return None
        return round(num / den, 4)

    tool_hits = sum(1 for item in tool_items if item.tool_expectation_hit is True)
    grounding_hits = sum(1 for item in grounding_items if item.grounding_pass is True)
    no_evidence_leaks = sum(1 for item in no_evidence_items if item.no_evidence_answer_leakage is True)

    return ModeSummary(
        mode=mode,
        total_cases=total,
        request_success_rate=round(request_success_count / total, 4) if total else 0.0,
        tool_expectation_hit_rate=_ratio(tool_hits, len(tool_items)),
        grounding_pass_rate=_ratio(grounding_hits, len(grounding_items)),
        no_evidence_answer_rate=_ratio(no_evidence_leaks, len(no_evidence_items)),
        request_success_count=request_success_count,
        tool_expectation_denominator=len(tool_items),
        grounding_denominator=len(grounding_items),
        no_evidence_denominator=len(no_evidence_items),
    )


def _write_results_csv(path: Path, items: list[ScoredOutcome]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "case_id",
                "mode",
                "endpoint",
                "request_success",
                "status_code",
                "duration_ms",
                "tool_expectation_applicable",
                "tool_expectation_hit",
                "grounding_applicable",
                "grounding_pass",
                "no_evidence_applicable",
                "no_evidence_answer_leakage",
                "expected_tools",
                "observed_tools",
                "route",
                "collection_name",
                "phase",
                "trace_ref",
                "validation_passed",
                "citations_verified",
                "sources_count",
                "error",
            ],
        )
        writer.writeheader()
        for item in items:
            writer.writerow(
                {
                    "case_id": item.case_id,
                    "mode": item.mode,
                    "endpoint": item.endpoint,
                    "request_success": item.request_success,
                    "status_code": item.status_code,
                    "duration_ms": item.duration_ms,
                    "tool_expectation_applicable": item.tool_expectation_applicable,
                    "tool_expectation_hit": item.tool_expectation_hit,
                    "grounding_applicable": item.grounding_applicable,
                    "grounding_pass": item.grounding_pass,
                    "no_evidence_applicable": item.no_evidence_applicable,
                    "no_evidence_answer_leakage": item.no_evidence_answer_leakage,
                    "expected_tools": ",".join(item.expected_tools),
                    "observed_tools": ",".join(item.observed_tools),
                    "route": item.route,
                    "collection_name": item.collection_name,
                    "phase": item.phase or "",
                    "trace_ref": item.trace_ref or "",
                    "validation_passed": item.validation_passed,
                    "citations_verified": item.citations_verified,
                    "sources_count": item.sources_count,
                    "error": item.error or "",
                }
            )
