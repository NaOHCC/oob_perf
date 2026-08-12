# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Compare two completed callable performance analyses offline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from perf_analysis.comparison import (
    ComparisonError,
    compare_analyses,
    load_analysis,
    load_reference,
)
from perf_analysis.reporting import (
    render_comparison_text,
    write_comparison_json,
    write_comparison_markdown,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "reference",
        type=Path,
        help="Reference XPU analysis.json or collection.json",
    )
    parser.add_argument("target", type=Path, help="Target analysis.json")
    parser.add_argument(
        "--reference-unitrace",
        type=Path,
        help="Explicit unitrace JSON for an XPU collection reference",
    )
    parser.add_argument(
        "--allow-reference-profiler-fallback",
        action="store_true",
        help="Use XPU profiler durations when reference unitrace analysis fails",
    )
    parser.add_argument("--reference-name", help="Reference display name")
    parser.add_argument("--target-name", help="Target display name")
    parser.add_argument(
        "--top-operations",
        type=int,
        default=20,
        help="Number of operator gaps in terminal and Markdown output",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(),
        help="Directory for comparison.json and comparison.md",
    )
    return parser


def main() -> None:
    """Run the offline cross-device comparison phase."""
    parser = _parser()
    args = parser.parse_args()
    if args.top_operations <= 0:
        parser.error("--top-operations must be greater than zero")
    try:
        result = compare_analyses(
            load_reference(
                args.reference,
                unitrace_path=args.reference_unitrace,
                allow_profiler_fallback=args.allow_reference_profiler_fallback,
            ),
            load_analysis(args.target),
            reference_name=args.reference_name,
            target_name=args.target_name,
        )
    except ComparisonError as error:
        parser.error(str(error))
    write_comparison_json(result, args.output_dir / "comparison.json")
    write_comparison_markdown(
        result,
        args.output_dir / "comparison.md",
        top_operations=args.top_operations,
    )
    sys.stdout.write(
        render_comparison_text(result, top_operations=args.top_operations) + "\n",
    )


if __name__ == "__main__":
    main()
