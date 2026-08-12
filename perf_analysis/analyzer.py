# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Combine collection manifests and device traces into roofline results."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from perf_analysis.models import (
    SCHEMA_VERSION,
    ActualSource,
    AnalysisResult,
    CollectionArtifacts,
    OperatorAnalysis,
)
from perf_analysis.traces import (
    ActualOp,
    TraceAnalysisError,
    normalize_op_name,
    parse_profiler_ops,
    parse_unitrace_ops,
)

_VECTOR_ENGINE_OPS = frozenset(
    {
        "aten::_softmax",
        "aten::max_pool2d_with_indices",
        "aten::native_batch_norm",
        "aten::native_layer_norm",
    },
)
_EXCLUDED_OPS = frozenset({"__view_noop__", "aten::copy_"})


@dataclass
class _ProjectedAggregate:
    projected_ms: float = 0.0
    flops: int = 0
    memory_bytes: int = 0
    calls: int = 0
    compute_bound_calls: int = 0
    memory_bound_calls: int = 0


@dataclass
class _ActualAggregate:
    duration_us: float = 0.0
    calls: int = 0
    shape_times: dict[tuple[str, str], float] = field(default_factory=dict)


def load_collection(path: Path) -> CollectionArtifacts:
    """Load and validate a collection manifest.

    Returns:
        The deserialized collection.

    Raises:
        TraceAnalysisError: If the manifest is missing or malformed.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        msg = f"invalid collection manifest: {path}"
        raise TraceAnalysisError(msg) from error
    if not isinstance(data, dict):
        msg = f"collection manifest must contain an object: {path}"
        raise TraceAnalysisError(msg)
    return CollectionArtifacts.from_dict(data)


def _resolve_unitrace_path(
    collection: CollectionArtifacts,
    explicit_path: Path | None,
) -> Path:
    if explicit_path is not None:
        if not explicit_path.is_file():
            msg = f"unitrace artifact does not exist: {explicit_path}"
            raise TraceAnalysisError(msg)
        return explicit_path
    pattern = Path(collection.unitrace_glob)
    matches = list(pattern.parent.glob(pattern.name))
    if len(matches) != 1:
        msg = f"expected one unitrace artifact matching {collection.unitrace_glob!r}, found {len(matches)}"
        raise TraceAnalysisError(msg)
    return matches[0]


def _project_ops(
    collection: CollectionArtifacts,
) -> dict[str, _ProjectedAggregate]:
    projected: dict[str, _ProjectedAggregate] = {}
    hardware = collection.hardware
    for invocation in collection.invocations:
        name = normalize_op_name(invocation.name)
        if name in _EXCLUDED_OPS:
            continue
        projection_flops = 0 if name in _VECTOR_ENGINE_OPS else invocation.flops
        compute_ms = projection_flops / hardware.peak_flops_per_second * 1000
        memory_ms = invocation.memory_bytes / hardware.memory_bytes_per_second * 1000
        projected_ms = max(compute_ms, memory_ms)
        aggregate = projected.setdefault(name, _ProjectedAggregate())
        aggregate.projected_ms += projected_ms
        aggregate.flops += invocation.flops
        aggregate.memory_bytes += invocation.memory_bytes
        aggregate.calls += 1
        if projected_ms > 0 and compute_ms >= memory_ms:
            aggregate.compute_bound_calls += 1
        elif projected_ms > 0:
            aggregate.memory_bound_calls += 1
    return projected


def _aggregate_actual(operations: list[ActualOp]) -> dict[str, _ActualAggregate]:
    actual: dict[str, _ActualAggregate] = {}
    for operation in operations:
        if operation.name in _EXCLUDED_OPS:
            continue
        aggregate = actual.setdefault(operation.name, _ActualAggregate())
        aggregate.duration_us += operation.gpu_duration_us
        aggregate.calls += 1
        shape = (operation.input_dims, operation.input_strides)
        aggregate.shape_times[shape] = (
            aggregate.shape_times.get(shape, 0.0) + operation.gpu_duration_us
        )
    return actual


def _operator_results(
    projected: dict[str, _ProjectedAggregate],
    actual: dict[str, _ActualAggregate],
) -> list[OperatorAnalysis]:
    results = []
    for name in projected.keys() | actual.keys():
        projection = projected.get(name, _ProjectedAggregate())
        measured = actual.get(name)
        dominant_shape = ("", "")
        if measured is not None and measured.shape_times:
            dominant_shape = max(
                measured.shape_times,
                key=lambda shape: measured.shape_times[shape],
            )
        results.append(
            OperatorAnalysis(
                name=name,
                projected_ms=projection.projected_ms,
                actual_ms=None if measured is None else measured.duration_us / 1000,
                flops=projection.flops,
                memory_bytes=projection.memory_bytes,
                projected_calls=projection.calls,
                actual_calls=0 if measured is None else measured.calls,
                compute_bound_calls=projection.compute_bound_calls,
                memory_bound_calls=projection.memory_bound_calls,
                dominant_shape=dominant_shape[0],
                dominant_stride=dominant_shape[1],
            ),
        )
    return sorted(
        results,
        key=lambda item: (
            -1 if item.actual_ms is None else item.actual_ms,
            item.projected_ms,
        ),
        reverse=True,
    )


def analyze_collection(
    collection: CollectionArtifacts | Path,
    *,
    unitrace_path: Path | None = None,
    allow_profiler_fallback: bool = False,
) -> AnalysisResult:
    """Analyze one collection after its unitrace-wrapped process exits.

    Args:
        collection: In-memory manifest or path to ``collection.json``.
        unitrace_path: Explicit unitrace Chrome trace. The manifest glob is
            used when omitted.
        allow_profiler_fallback: Use profiler device durations when unitrace is
            missing or cannot be mapped strictly.

    Returns:
        Structured end-to-end and per-operator roofline results.

    Raises:
        TraceAnalysisError: If strict unitrace analysis cannot be completed.
    """
    manifest = (
        load_collection(collection) if isinstance(collection, Path) else collection
    )
    diagnostics = [
        f"unaccounted FLOPs: {name} ({calls} calls)"
        for name, calls in sorted(manifest.unaccounted_flop_ops.items())
    ]
    source_preference = manifest.actual_source_preference
    if source_preference is None:
        source_preference = "unitrace" if manifest.device == "xpu" else "profiler"
    actual_source: ActualSource = source_preference
    if source_preference == "profiler":
        actual_operations = parse_profiler_ops(manifest.trace_path)
    else:
        try:
            resolved_unitrace = _resolve_unitrace_path(manifest, unitrace_path)
            actual_operations = parse_unitrace_ops(
                manifest.trace_path,
                resolved_unitrace,
                diagnostics=diagnostics,
            )
        except TraceAnalysisError as error:
            if not allow_profiler_fallback:
                raise
            actual_source = "profiler"
            diagnostics.append(f"unitrace fallback: {error}")
            actual_operations = parse_profiler_ops(manifest.trace_path)

    projected = _project_ops(manifest)
    actual = _aggregate_actual(actual_operations)
    operators = _operator_results(projected, actual)
    t1_ms = sum(item.projected_ms for item in projected.values())
    device_duration_ms = sum(item.duration_us for item in actual.values()) / 1000
    return AnalysisResult(
        schema_version=SCHEMA_VERSION,
        workload_name=manifest.workload_name,
        device=manifest.device,
        hardware=manifest.hardware,
        latency=manifest.latency,
        t1_ms=t1_ms,
        t2_wall_ms=manifest.latency.median_ms,
        t2_device_ms=device_duration_ms if device_duration_ms > 0 else None,
        actual_source=actual_source,
        totals=manifest.totals,
        operators=operators,
        diagnostics=diagnostics,
    )
