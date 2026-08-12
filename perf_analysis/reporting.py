# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Render and persist single-workload performance analysis results."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from perf_analysis.models import (
        AnalysisResult,
        ComparisonResult,
        OperatorAnalysis,
        OperatorComparison,
    )


def _format_optional(value: float | None, suffix: str = "") -> str:
    return "n/a" if value is None else f"{value:.4f}{suffix}"


def _bound(operator: OperatorAnalysis) -> str:
    if operator.compute_bound_calls > operator.memory_bound_calls:
        return "compute"
    if operator.memory_bound_calls > 0:
        return "memory"
    return "none"


def _device_wall_ratio(result: AnalysisResult) -> float | None:
    if result.t2_device_ms is None or result.t2_wall_ms <= 0:
        return None
    return result.t2_device_ms / result.t2_wall_ms


def _comparison_rows(
    result: ComparisonResult,
    top_operations: int | None,
) -> list[OperatorComparison]:
    return (
        result.operators
        if top_operations is None
        else result.operators[:top_operations]
    )


def render_text(result: AnalysisResult, *, top_operations: int = 20) -> str:
    """Render a compact terminal report.

    Returns:
        Multi-line text suitable for terminal output.
    """
    lines = [
        f"Performance analysis: {result.workload_name}",
        f"  Device: {result.device}",
        f"  Hardware: {result.hardware.label}",
        f"  T1 projection: {result.t1_ms:.4f} ms",
        f"  T2 wall median: {result.t2_wall_ms:.4f} ms",
        f"  T2 device sum: {_format_optional(result.t2_device_ms, ' ms')}",
        f"  R = T1/T2 wall: {result.efficiency:.4f}",
        f"  Actual source: {result.actual_source}",
        "  Top operators:",
    ]
    lines.extend(
        "".join(
            (
                f"    {operator.name}: T1={operator.projected_ms:.4f} ms, ",
                f"actual={_format_optional(operator.actual_ms, ' ms')}, ",
                f"R={_format_optional(operator.efficiency)}, bound={_bound(operator)}",
            ),
        )
        for operator in result.operators[:top_operations]
    )
    if result.diagnostics:
        lines.append("  Diagnostics:")
        lines.extend(f"    {message}" for message in result.diagnostics)
    return "\n".join(lines)


def render_markdown(result: AnalysisResult) -> str:
    """Render a complete single-workload Markdown report.

    Returns:
        Markdown containing summaries, operator metrics, and diagnostics.
    """
    totals = result.totals
    operator_header = "| Operator | Bound | Calls (projected/actual) | T1 (ms) | Actual (ms) | R | FLOPs | Logical bytes | Dominant shape |"  # noqa: E501
    lines = [
        f"# Performance Analysis: {result.workload_name}",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Device | {result.device} |",
        f"| Actual source | {result.actual_source} |",
        f"| T1 projection | {result.t1_ms:.4f} ms |",
        f"| T2 wall median | {result.t2_wall_ms:.4f} ms |",
        f"| T2 device sum | {_format_optional(result.t2_device_ms, ' ms')} |",
        f"| R = T1 / T2 wall | {result.efficiency:.4f} |",
        f"| R device diagnostic | {_format_optional(result.device_efficiency)} |",
        "",
        "## Hardware",
        "",
        "| Specification | Value |",
        "|---|---:|",
        f"| Label | {result.hardware.label} |",
        f"| Peak throughput | {result.hardware.peak_tflops:.4f} TFLOPS |",
        f"| DRAM bandwidth | {result.hardware.memory_bandwidth_gbs:.4f} GB/s |",
        "",
        "## Workload Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| FLOPs | {totals.get('flops', 0)} |",
        f"| Exact FLOPs | {totals.get('exact_flops', 0)} |",
        f"| Estimated FLOPs | {totals.get('estimated_flops', 0)} |",
        f"| GEMM/attention FLOPs | {totals.get('gemm_attention_flops', 0)} |",
        f"| Logical data movement | {totals.get('memory_bytes', 0)} B |",
        "",
        "## Per-Operator Comparison",
        "",
        operator_header,
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    lines.extend(
        "".join(
            (
                f"| {operator.name} | {_bound(operator)} | ",
                f"{operator.projected_calls}/{operator.actual_calls} | ",
                f"{operator.projected_ms:.4f} | {_format_optional(operator.actual_ms)} | ",
                f"{_format_optional(operator.efficiency)} | {operator.flops} | ",
                f"{operator.memory_bytes} | {operator.dominant_shape} |",
            ),
        )
        for operator in result.operators
    )
    lines.extend(["", "## Diagnostics", ""])
    if result.diagnostics:
        lines.extend(f"- {message}" for message in result.diagnostics)
    else:
        lines.append("No collection or mapping diagnostics.")
    lines.append("")
    return "\n".join(lines)


def render_comparison_text(
    result: ComparisonResult,
    *,
    top_operations: int = 20,
) -> str:
    """Render a compact terminal cross-device comparison.

    Returns:
        Multi-line text suitable for terminal output.
    """
    reference = result.reference
    target = result.target
    lines = [
        f"Performance comparison: {result.workload_name}",
        f"  Reference: {result.reference_name} ({reference.actual_source})",
        f"  Target: {result.target_name} ({target.actual_source})",
        f"  Comparable workload: {'yes' if result.comparable else 'no'}",
        f"  Target wall speedup: {result.wall_speedup:.4f}x",
        f"  Target p95 speedup: {result.p95_speedup:.4f}x",
        "  End-to-end:",
        (
            f"    {result.reference_name}: median={reference.t2_wall_ms:.4f} ms, "
            f"p95={reference.latency.p95_ms:.4f} ms, T1={reference.t1_ms:.4f} ms, "
            f"R={reference.efficiency:.4f}"
        ),
        (
            f"    {result.target_name}: median={target.t2_wall_ms:.4f} ms, "
            f"p95={target.latency.p95_ms:.4f} ms, T1={target.t1_ms:.4f} ms, "
            f"R={target.efficiency:.4f}"
        ),
        "  Top operator gaps (diagnostic):",
    ]
    lines.extend(
        (
            f"    {operator.name}: reference="
            f"{_format_optional(operator.reference_actual_ms, ' ms')}, target="
            f"{_format_optional(operator.target_actual_ms, ' ms')}, speedup="
            f"{_format_optional(operator.target_speedup, 'x')}, gap="
            f"{_format_optional(operator.reference_minus_target_ms, ' ms')}"
        )
        for operator in _comparison_rows(result, top_operations)
    )
    if result.warnings:
        lines.append("  Comparability warnings:")
        lines.extend(f"    {warning}" for warning in result.warnings)
    return "\n".join(lines)


def render_comparison_markdown(
    result: ComparisonResult,
    *,
    top_operations: int | None = None,
) -> str:
    """Render a complete cross-device Markdown report.

    Returns:
        Markdown containing end-to-end, operator, and diagnostic comparisons.
    """
    reference = result.reference
    target = result.target
    lines = [
        f"# Performance Comparison: {result.workload_name}",
        "",
        "## Summary",
        "",
        "| Metric | Reference | Target | Target / reference |",
        "|---|---:|---:|---:|",
        f"| Device | {result.reference_name} | {result.target_name} | n/a |",
        f"| Actual source | {reference.actual_source} | {target.actual_source} | n/a |",
        (
            f"| Wall median | {reference.t2_wall_ms:.4f} ms | "
            f"{target.t2_wall_ms:.4f} ms | {result.wall_speedup:.4f}x speedup |"
        ),
        (
            f"| Wall p95 | {reference.latency.p95_ms:.4f} ms | "
            f"{target.latency.p95_ms:.4f} ms | {result.p95_speedup:.4f}x speedup |"
        ),
        f"| T1 projection | {reference.t1_ms:.4f} ms | {target.t1_ms:.4f} ms | n/a |",
        f"| R = T1 / T2 wall | {reference.efficiency:.4f} | {target.efficiency:.4f} | n/a |",
        (
            f"| T2 device sum | {_format_optional(reference.t2_device_ms, ' ms')} | "
            f"{_format_optional(target.t2_device_ms, ' ms')} | diagnostic |"
        ),
        (
            f"| Device / wall ratio | {_format_optional(_device_wall_ratio(reference))} | "
            f"{_format_optional(_device_wall_ratio(target))} | diagnostic |"
        ),
        (
            f"| Comparable workload | {'yes' if result.comparable else 'no'} | "
            f"{'yes' if result.comparable else 'no'} | n/a |"
        ),
        "",
        (
            "Target speedup is reference wall time divided by target wall time. "
            "Per-device R uses each device's own hardware roofline."
        ),
        "",
        "## Hardware",
        "",
        "| Specification | Reference | Target |",
        "|---|---:|---:|",
        f"| Label | {reference.hardware.label} | {target.hardware.label} |",
        (
            f"| Peak throughput | {reference.hardware.peak_tflops:.4f} TFLOPS | "
            f"{target.hardware.peak_tflops:.4f} TFLOPS |"
        ),
        (
            f"| DRAM bandwidth | {reference.hardware.memory_bandwidth_gbs:.4f} GB/s | "
            f"{target.hardware.memory_bandwidth_gbs:.4f} GB/s |"
        ),
        "",
        "## Per-Operator Diagnostics",
        "",
        (
            "Per-operator timings can use different trace sources. Use this table "
            "for diagnosis, not as the end-to-end performance conclusion."
        ),
        "",
        (
            "| Operator | Reference actual (ms) | Target actual (ms) | Target speedup "
            "| Reference - target (ms) | Reference R | Target R | "
            "Bound (reference/target) | Calls (reference/target) |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    lines.extend(
        (
            f"| {operator.name} | "
            f"{_format_optional(operator.reference_actual_ms)} | "
            f"{_format_optional(operator.target_actual_ms)} | "
            f"{_format_optional(operator.target_speedup, 'x')} | "
            f"{_format_optional(operator.reference_minus_target_ms)} | "
            f"{_format_optional(operator.reference_efficiency)} | "
            f"{_format_optional(operator.target_efficiency)} | "
            f"{operator.reference_bound or 'n/a'}/{operator.target_bound or 'n/a'} | "
            f"{operator.reference_calls if operator.reference_calls is not None else 'n/a'}/"
            f"{operator.target_calls if operator.target_calls is not None else 'n/a'} |"
        )
        for operator in _comparison_rows(result, top_operations)
    )
    lines.extend(["", "## Comparability", ""])
    if result.warnings:
        lines.extend(f"- {warning}" for warning in result.warnings)
    else:
        lines.append("Workload metrics and projected calls match.")
    lines.extend(["", f"## Diagnostics: {result.reference_name}", ""])
    if reference.diagnostics:
        lines.extend(f"- {message}" for message in reference.diagnostics)
    else:
        lines.append("No collection or mapping diagnostics.")
    lines.extend(["", f"## Diagnostics: {result.target_name}", ""])
    if target.diagnostics:
        lines.extend(f"- {message}" for message in target.diagnostics)
    else:
        lines.append("No collection or mapping diagnostics.")
    lines.append("")
    return "\n".join(lines)


def write_json(result: AnalysisResult, path: Path) -> Path:
    """Write a structured JSON report.

    Returns:
        The output path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def write_markdown(result: AnalysisResult, path: Path) -> Path:
    """Write a Markdown report.

    Returns:
        The output path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown(result), encoding="utf-8")
    return path


def write_comparison_json(result: ComparisonResult, path: Path) -> Path:
    """Write a structured comparison JSON report.

    Returns:
        The output path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def write_comparison_markdown(
    result: ComparisonResult,
    path: Path,
    *,
    top_operations: int | None = None,
) -> Path:
    """Write a comparison Markdown report.

    Returns:
        The output path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        render_comparison_markdown(result, top_operations=top_operations),
        encoding="utf-8",
    )
    return path
