# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Compare completed single-device performance analyses offline."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from perf_analysis.models import (
    AnalysisResult,
    ComparisonResult,
    OperatorAnalysis,
    OperatorComparison,
    SCHEMA_VERSION,
)

if TYPE_CHECKING:
    from pathlib import Path

_COMPARABILITY_TOTALS = (
    "flops",
    "exact_flops",
    "estimated_flops",
    "gemm_attention_flops",
    "memory_bytes",
)


class ComparisonError(RuntimeError):
    """Raised when analysis results cannot form a valid comparison."""


def load_analysis(path: Path) -> AnalysisResult:
    """Load a structured single-device analysis result.

    Returns:
        The deserialized analysis result.

    Raises:
        ComparisonError: If the report is missing, malformed, or unsupported.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        msg = f"invalid analysis report: {path}"
        raise ComparisonError(msg) from error
    if not isinstance(data, dict):
        msg = f"analysis JSON must contain an object: {path}"
        raise ComparisonError(msg)
    try:
        return AnalysisResult.from_dict(data)
    except (KeyError, TypeError, ValueError) as error:
        msg = f"invalid analysis report: {path}"
        raise ComparisonError(msg) from error


def _bound(operator: OperatorAnalysis) -> str:
    if operator.compute_bound_calls > operator.memory_bound_calls:
        return "compute"
    if operator.memory_bound_calls > 0:
        return "memory"
    return "none"


def _positive_ratio(numerator: float, denominator: float) -> float | None:
    if numerator <= 0 or denominator <= 0:
        return None
    return numerator / denominator


def _operator_comparison(
    name: str,
    reference: OperatorAnalysis | None,
    target: OperatorAnalysis | None,
) -> OperatorComparison:
    reference_actual = None if reference is None else reference.actual_ms
    target_actual = None if target is None else target.actual_ms
    speedup = None
    gap = None
    if reference_actual is not None and target_actual is not None:
        speedup = _positive_ratio(reference_actual, target_actual)
        gap = reference_actual - target_actual
    return OperatorComparison(
        name=name,
        reference_actual_ms=reference_actual,
        target_actual_ms=target_actual,
        target_speedup=speedup,
        reference_minus_target_ms=gap,
        reference_projected_ms=(None if reference is None else reference.projected_ms),
        target_projected_ms=None if target is None else target.projected_ms,
        reference_efficiency=None if reference is None else reference.efficiency,
        target_efficiency=None if target is None else target.efficiency,
        reference_calls=None if reference is None else reference.actual_calls,
        target_calls=None if target is None else target.actual_calls,
        reference_bound=None if reference is None else _bound(reference),
        target_bound=None if target is None else _bound(target),
    )


def compare_analyses(
    reference: AnalysisResult,
    target: AnalysisResult,
    *,
    reference_name: str | None = None,
    target_name: str | None = None,
) -> ComparisonResult:
    """Compare target performance against a reference result.

    Speedups use ``reference / target``. With XPU as the reference and A100 as
    the target, values greater than one mean A100 is faster.

    Returns:
        End-to-end and per-operator comparison metrics.

    Raises:
        ComparisonError: If workload names differ or wall timings are invalid.
    """
    if reference.workload_name != target.workload_name:
        msg = (
            f"workload mismatch: {reference.workload_name!r} != "
            f"{target.workload_name!r}"
        )
        raise ComparisonError(msg)
    wall_speedup = _positive_ratio(reference.t2_wall_ms, target.t2_wall_ms)
    p95_speedup = _positive_ratio(reference.latency.p95_ms, target.latency.p95_ms)
    if wall_speedup is None or p95_speedup is None:
        msg = "wall median and p95 latencies must be positive"
        raise ComparisonError(msg)

    warnings = []
    for key in _COMPARABILITY_TOTALS:
        reference_value = reference.totals.get(key, 0)
        target_value = target.totals.get(key, 0)
        if reference_value != target_value:
            warnings.append(
                f"workload metric mismatch for {key}: "
                f"reference={reference_value}, target={target_value}",
            )

    reference_ops = {operator.name: operator for operator in reference.operators}
    target_ops = {operator.name: operator for operator in target.operators}
    for name in sorted(reference_ops.keys() & target_ops.keys()):
        reference_calls = reference_ops[name].projected_calls
        target_calls = target_ops[name].projected_calls
        if reference_calls != target_calls:
            warnings.append(
                f"projected call mismatch for {name}: "
                f"reference={reference_calls}, target={target_calls}",
            )

    operators = [
        _operator_comparison(name, reference_ops.get(name), target_ops.get(name))
        for name in reference_ops.keys() | target_ops.keys()
    ]
    operators.sort(
        key=lambda operator: (
            operator.reference_minus_target_ms is not None,
            operator.reference_minus_target_ms or 0.0,
        ),
        reverse=True,
    )
    return ComparisonResult(
        schema_version=SCHEMA_VERSION,
        workload_name=reference.workload_name,
        reference_name=reference_name or reference.hardware.label,
        target_name=target_name or target.hardware.label,
        reference=reference,
        target=target,
        wall_speedup=wall_speedup,
        p95_speedup=p95_speedup,
        operators=operators,
        comparable=not warnings,
        warnings=warnings,
    )
