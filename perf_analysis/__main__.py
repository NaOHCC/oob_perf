# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Generate reports from a completed callable performance collection."""

from __future__ import annotations

import argparse
from pathlib import Path

from perf_analysis.analyzer import analyze_collection_artifacts
from perf_analysis.reporting import (
    render_text,
    write_calls_json,
    write_json,
    write_markdown,
)
from perf_analysis.traces import TraceAnalysisError


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("collection", type=Path, help="Path to collection.json")
    parser.add_argument("--unitrace", type=Path, help="Explicit unitrace JSON path")
    parser.add_argument(
        "--allow-profiler-fallback",
        action="store_true",
        help="Use profiler durations when strict unitrace analysis fails",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(),
        help="Directory for analysis.json and analysis.md",
    )
    return parser


def main() -> None:
    """Run the offline report phase."""
    parser = _parser()
    args = parser.parse_args()
    try:
        result, calls = analyze_collection_artifacts(
            args.collection,
            unitrace_path=args.unitrace,
            allow_profiler_fallback=args.allow_profiler_fallback,
        )
    except TraceAnalysisError as error:
        parser.error(str(error))
    write_json(result, args.output_dir / "analysis.json")
    write_markdown(result, args.output_dir / "analysis.md")
    write_calls_json(calls, args.output_dir / "calls.json")
    print(render_text(result))  # noqa: T201


if __name__ == "__main__":
    main()
