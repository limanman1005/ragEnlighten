"""Run baseline offline regression evaluation for React and Plan-Execute agents."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evals.baseline import (
    PLAN_EXECUTE_MODE,
    REACT_MODE,
    build_summary,
    load_cases,
    run_cases_http,
    score_outcomes,
    write_results,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="evals/baseline_cases.jsonl",
        help="Path to evaluation dataset file (jsonl).",
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="FastAPI base URL for chat endpoints.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=30.0,
        help="HTTP timeout for each case request.",
    )
    parser.add_argument(
        "--mode",
        dest="modes",
        action="append",
        choices=[REACT_MODE, PLAN_EXECUTE_MODE],
        help="Evaluation mode to run. Repeat to run multiple modes. Defaults to both.",
    )
    parser.add_argument(
        "--output-dir",
        default="evals/results/latest",
        help="Directory for results.json/summary.json and optional csv export.",
    )
    parser.add_argument(
        "--export-csv",
        action="store_true",
        help="Export per-case results.csv for manual review.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    modes = args.modes or [REACT_MODE, PLAN_EXECUTE_MODE]
    dataset_path = Path(args.dataset)
    cases = load_cases(dataset_path)

    outcomes = run_cases_http(
        cases,
        base_url=args.base_url,
        timeout_seconds=args.timeout_seconds,
        modes=modes,
    )
    scored = score_outcomes(outcomes)
    summary = build_summary(scored)

    output_paths = write_results(
        output_dir=args.output_dir,
        scored_outcomes=scored,
        summary=summary,
        export_csv=args.export_csv,
    )

    print(f"Dataset: {dataset_path}")
    print(f"Modes: {', '.join(modes)}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("Artifacts:")
    for key, value in output_paths.items():
        print(f"- {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

