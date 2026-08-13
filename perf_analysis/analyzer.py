# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Combine collection manifests and device traces into roofline results."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from perf_analysis.models import (
    ActualSource,
    AnalysisResult,
    CallDataset,
    CallRecord,
    CALLS_SCHEMA_VERSION,
    CollectionArtifacts,
    OperatorAnalysis,
    SCHEMA_VERSION,
)
from perf_analysis.traces import (
    ActualOp,
    normalize_op_name,
    parse_profiler_ops,
    parse_unitrace_ops,
    TraceAnalysisError,
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


@dataclass(frozen=True)
class _ProjectedCall:
    name: str
    raw_name: str
    source_index: int
    projected_ms: float
    compute_ms: float
    memory_ms: float
    flops: int
    memory_bytes: int
    bound: str


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


def _project_calls(collection: CollectionArtifacts) -> list[_ProjectedCall]:
    calls = []
    hardware = collection.hardware
    for source_index, invocation in enumerate(collection.invocations):
        name = normalize_op_name(invocation.name)
        if name in _EXCLUDED_OPS:
            continue
        projection_flops = 0 if name in _VECTOR_ENGINE_OPS else invocation.flops
        compute_ms = projection_flops / hardware.peak_flops_per_second * 1000
        memory_ms = invocation.memory_bytes / hardware.memory_bytes_per_second * 1000
        projected_ms = max(compute_ms, memory_ms)
        bound = "none"
        if projected_ms > 0:
            bound = "compute" if compute_ms >= memory_ms else "memory"
        calls.append(
            _ProjectedCall(
                name=name,
                raw_name=invocation.name,
                source_index=source_index,
                projected_ms=projected_ms,
                compute_ms=compute_ms,
                memory_ms=memory_ms,
                flops=invocation.flops,
                memory_bytes=invocation.memory_bytes,
                bound=bound,
            ),
        )
    return calls


def _aggregate_projected(
    calls: list[_ProjectedCall],
) -> dict[str, _ProjectedAggregate]:
    projected: dict[str, _ProjectedAggregate] = {}
    for call in calls:
        aggregate = projected.setdefault(call.name, _ProjectedAggregate())
        aggregate.projected_ms += call.projected_ms
        aggregate.flops += call.flops
        aggregate.memory_bytes += call.memory_bytes
        aggregate.calls += 1
        if call.bound == "compute":
            aggregate.compute_bound_calls += 1
        elif call.bound == "memory":
            aggregate.memory_bound_calls += 1
    return projected


def _project_ops(
    collection: CollectionArtifacts,
) -> dict[str, _ProjectedAggregate]:
    return _aggregate_projected(_project_calls(collection))


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


def _calls_dataset(
    collection: CollectionArtifacts,
    actual_source: ActualSource,
    projected_calls: list[_ProjectedCall],
    actual_operations: list[ActualOp],
) -> CallDataset:
    projected_by_name: dict[str, list[_ProjectedCall]] = {}
    for call in projected_calls:
        projected_by_name.setdefault(call.name, []).append(call)

    actual_by_name: dict[str, list[ActualOp]] = {}
    for operation in actual_operations:
        if operation.name not in _EXCLUDED_OPS:
            actual_by_name.setdefault(operation.name, []).append(operation)

    records = []
    for name in sorted(projected_by_name.keys() | actual_by_name.keys()):
        projections = projected_by_name.get(name, [])
        actuals = actual_by_name.get(name, [])
        for ordinal in range(max(len(projections), len(actuals))):
            projection = projections[ordinal] if ordinal < len(projections) else None
            actual = actuals[ordinal] if ordinal < len(actuals) else None
            match_status = (
                "sequence-paired"
                if projection is not None and actual is not None
                else "projected-only" if projection is not None else "actual-only"
            )
            records.append(
                CallRecord(
                    name=name,
                    match_status=match_status,
                    projected_raw_name=(
                        None if projection is None else projection.raw_name
                    ),
                    actual_raw_name=None if actual is None else actual.raw_name,
                    projected_index=(
                        None if projection is None else projection.source_index
                    ),
                    actual_index=None if actual is None else actual.source_index,
                    external_id=None if actual is None else actual.external_id,
                    projected_ms=(
                        None if projection is None else projection.projected_ms
                    ),
                    compute_ms=None if projection is None else projection.compute_ms,
                    memory_ms=None if projection is None else projection.memory_ms,
                    flops=None if projection is None else projection.flops,
                    memory_bytes=(
                        None if projection is None else projection.memory_bytes
                    ),
                    bound=None if projection is None else projection.bound,
                    actual_ms=None if actual is None else actual.gpu_duration_us / 1000,
                    timestamp_us=None if actual is None else actual.timestamp_us,
                    input_dims=None if actual is None else actual.input_dims,
                    input_strides=None if actual is None else actual.input_strides,
                ),
            )
    return CallDataset(
        schema_version=CALLS_SCHEMA_VERSION,
        workload_name=collection.workload_name,
        device=collection.device,
        actual_source=actual_source,
        calls=records,
    )


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


def analyze_collection_artifacts(
    collection: CollectionArtifacts | Path,
    *,
    unitrace_path: Path | None = None,
    allow_profiler_fallback: bool = False,
) -> tuple[AnalysisResult, CallDataset]:
    """Analyze one collection after its unitrace-wrapped process exits.

    Args:
        collection: In-memory manifest or path to ``collection.json``.
        unitrace_path: Explicit unitrace Chrome trace. The manifest glob is
            used when omitted.
        allow_profiler_fallback: Use profiler device durations when unitrace is
            missing or cannot be mapped strictly.

    Returns:
        Aggregate roofline results and a per-call browser artifact.

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

    projected_calls = _project_calls(manifest)
    projected = _aggregate_projected(projected_calls)
    actual = _aggregate_actual(actual_operations)
    operators = _operator_results(projected, actual)
    t1_ms = sum(item.projected_ms for item in projected.values())
    device_duration_ms = sum(item.duration_us for item in actual.values()) / 1000
    result = AnalysisResult(
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
    return result, _calls_dataset(
        manifest,
        actual_source,
        projected_calls,
        actual_operations,
    )


def analyze_collection(
    collection: CollectionArtifacts | Path,
    *,
    unitrace_path: Path | None = None,
    allow_profiler_fallback: bool = False,
) -> AnalysisResult:
    """Analyze one collection while preserving the aggregate API.

    Returns:
        Structured end-to-end and per-operator roofline results.
    """
    return analyze_collection_artifacts(
        collection,
        unitrace_path=unitrace_path,
        allow_profiler_fallback=allow_profiler_fallback,
    )[0]
