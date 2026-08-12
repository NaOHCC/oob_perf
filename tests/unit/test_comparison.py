# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: S101

"""Tests for offline cross-device performance comparison."""

import json
from pathlib import Path

import pytest
from perf_analysis.comparison import (
    ComparisonError,
    compare_analyses,
    load_analysis,
    load_reference,
)
from perf_analysis.models import (
    AnalysisResult,
    CollectionArtifacts,
    HardwareSpec,
    InvocationMetrics,
    LatencyStats,
    OperatorAnalysis,
)


def _operator(
    name: str,
    *,
    projected_ms: float,
    actual_ms: float | None,
    projected_calls: int = 1,
) -> OperatorAnalysis:
    return OperatorAnalysis(
        name=name,
        projected_ms=projected_ms,
        actual_ms=actual_ms,
        flops=10,
        memory_bytes=20,
        projected_calls=projected_calls,
        actual_calls=projected_calls,
        compute_bound_calls=projected_calls,
        memory_bound_calls=0,
    )


def _result(
    *,
    workload_name: str = "tiny",
    device: str,
    label: str,
    actual_source: str,
    median_ms: float,
    p95_ms: float,
    operators: list[OperatorAnalysis],
    totals: dict[str, int] | None = None,
) -> AnalysisResult:
    return AnalysisResult(
        schema_version=1,
        workload_name=workload_name,
        device=device,
        hardware=HardwareSpec(label, 10.0, 100.0),
        latency=LatencyStats(
            [median_ms],
            median_ms,
            p95_ms,
            median_ms,
            p95_ms,
        ),
        t1_ms=1.0,
        t2_wall_ms=median_ms,
        t2_device_ms=median_ms / 2,
        actual_source=actual_source,  # type: ignore[arg-type]
        totals=totals or {"flops": 10, "memory_bytes": 20},
        operators=operators,
    )


def _write_xpu_collection(
    tmp_path: Path,
    *,
    source_preference: str | None = "unitrace",
    write_unitrace: bool = True,
) -> tuple[Path, Path]:
    profiler_path = tmp_path / "trace.json"
    profiler_path.write_text(
        json.dumps(
            {
                "traceEvents": [
                    {
                        "ph": "X",
                        "cat": "cpu_op",
                        "name": "aten::mm",
                        "ts": 10,
                        "dur": 20,
                        "args": {"External id": 7},
                    },
                    {
                        "ph": "X",
                        "cat": "kernel",
                        "name": "gemm_kernel",
                        "ts": 12,
                        "dur": 8,
                        "args": {"External id": 7},
                    },
                ],
            },
        ),
        encoding="utf-8",
    )
    unitrace_path = tmp_path / "python.1.json"
    if write_unitrace:
        unitrace_path.write_text(
            json.dumps(
                {
                    "traceEvents": [
                        {"ph": "X", "name": "gemm_kernel", "ts": 12, "dur": 6},
                    ],
                },
            ),
            encoding="utf-8",
        )
    collection = CollectionArtifacts(
        schema_version=1,
        workload_name="tiny",
        pid=1,
        device="xpu",
        hardware=HardwareSpec("XPU BF16", 10.0, 100.0),
        latency=LatencyStats([4.0], 4.0, 5.0, 4.0, 5.0),
        invocations=[InvocationMetrics("aten::mm", 10, 0, 10, 20)],
        totals={"flops": 10, "memory_bytes": 20},
        unaccounted_flop_ops={},
        trace_path=profiler_path,
        unitrace_glob=str(tmp_path / "python.*.json"),
        actual_source_preference=source_preference,  # type: ignore[arg-type]
    )
    collection_path = tmp_path / "collection.json"
    collection_path.write_text(
        json.dumps(collection.to_dict()),
        encoding="utf-8",
    )
    return collection_path, unitrace_path


def test_analysis_json_round_trip_recomputes_efficiency(tmp_path: Path) -> None:
    """Loading ignores serialized derived values and reconstructs the result."""
    result = _result(
        device="xpu",
        label="XPU BF16",
        actual_source="profiler",
        median_ms=4.0,
        p95_ms=5.0,
        operators=[_operator("aten::mm", projected_ms=1.0, actual_ms=2.0)],
    )
    data = result.to_dict()
    data["efficiency"] = 999.0
    data["operators"][0]["efficiency"] = 999.0
    path = tmp_path / "analysis.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    loaded = load_analysis(path)

    assert loaded.efficiency == pytest.approx(0.25)
    assert loaded.operators[0].efficiency == pytest.approx(0.5)
    assert loaded.actual_source == "profiler"


@pytest.mark.parametrize("source_preference", ["unitrace", None])
@pytest.mark.parametrize("explicit_path", [False, True])
def test_reference_collection_uses_unitrace_duration(
    tmp_path: Path,
    *,
    explicit_path: bool,
    source_preference: str | None,
) -> None:
    """Collection references use unitrace rather than profiler durations."""
    collection_path, unitrace_path = _write_xpu_collection(
        tmp_path,
        source_preference=source_preference,
    )

    reference = load_reference(
        collection_path,
        unitrace_path=unitrace_path if explicit_path else None,
    )

    assert reference.actual_source == "unitrace"
    assert reference.t2_device_ms == pytest.approx(0.006)
    assert reference.operators[0].actual_ms == pytest.approx(0.006)


def test_reference_collection_unitrace_is_strict_unless_fallback_allowed(
    tmp_path: Path,
) -> None:
    """Missing unitrace fails by default and reports an explicit fallback."""
    collection_path, _unitrace_path = _write_xpu_collection(
        tmp_path,
        write_unitrace=False,
    )

    with pytest.raises(ComparisonError, match="expected one unitrace"):
        load_reference(collection_path)

    reference = load_reference(collection_path, allow_profiler_fallback=True)

    assert reference.actual_source == "profiler"
    assert reference.operators[0].actual_ms == pytest.approx(0.008)
    assert any("unitrace fallback" in message for message in reference.diagnostics)


def test_reference_rejects_profiler_only_xpu_collection(tmp_path: Path) -> None:
    """Profiler-only collections cannot be retrofitted with unitrace data."""
    collection_path, unitrace_path = _write_xpu_collection(
        tmp_path,
        source_preference="profiler",
    )

    with pytest.raises(ComparisonError, match="captured in profiler-only mode"):
        load_reference(collection_path, unitrace_path=unitrace_path)


def test_reference_rejects_non_xpu_collection(tmp_path: Path) -> None:
    """Only XPU collections can be analyzed as unitrace references."""
    collection_path, unitrace_path = _write_xpu_collection(tmp_path)
    data = json.loads(collection_path.read_text(encoding="utf-8"))
    data["device"] = "cuda"
    collection_path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ComparisonError, match="must use XPU"):
        load_reference(collection_path, unitrace_path=unitrace_path)


def test_comparison_uses_reference_over_target_speedup_and_sorts_gaps() -> None:
    """A100 speedup and operator gaps consistently use XPU as reference."""
    reference = _result(
        device="xpu",
        label="XPU BF16",
        actual_source="unitrace",
        median_ms=8.0,
        p95_ms=10.0,
        operators=[
            _operator("aten::mm", projected_ms=2.0, actual_ms=6.0),
            _operator("aten::relu", projected_ms=1.0, actual_ms=2.0),
        ],
    )
    target = _result(
        device="cuda",
        label="A100 BF16",
        actual_source="profiler",
        median_ms=2.0,
        p95_ms=4.0,
        operators=[
            _operator("aten::mm", projected_ms=1.0, actual_ms=2.0),
            _operator("aten::relu", projected_ms=0.5, actual_ms=1.0),
        ],
    )

    comparison = compare_analyses(reference, target)

    assert comparison.wall_speedup == pytest.approx(4.0)
    assert comparison.p95_speedup == pytest.approx(2.5)
    assert [operator.name for operator in comparison.operators] == [
        "aten::mm",
        "aten::relu",
    ]
    assert comparison.operators[0].target_speedup == pytest.approx(3.0)
    assert comparison.operators[0].reference_minus_target_ms == pytest.approx(4.0)
    assert comparison.comparable


def test_comparison_preserves_missing_ops_and_reports_incompatible_metrics() -> None:
    """The operator union remains visible when workload metrics differ."""
    reference = _result(
        device="xpu",
        label="XPU BF16",
        actual_source="profiler",
        median_ms=8.0,
        p95_ms=9.0,
        operators=[_operator("aten::mm", projected_ms=2.0, actual_ms=6.0)],
        totals={"flops": 10, "memory_bytes": 20},
    )
    target = _result(
        device="cuda",
        label="A100 BF16",
        actual_source="profiler",
        median_ms=4.0,
        p95_ms=5.0,
        operators=[_operator("aten::relu", projected_ms=1.0, actual_ms=1.0)],
        totals={"flops": 11, "memory_bytes": 20},
    )

    comparison = compare_analyses(reference, target)

    by_name = {operator.name: operator for operator in comparison.operators}
    assert not comparison.comparable
    assert comparison.warnings == [
        "workload metric mismatch for flops: reference=10, target=11",
    ]
    assert by_name["aten::mm"].target_actual_ms is None
    assert by_name["aten::relu"].reference_actual_ms is None
    assert by_name["aten::mm"].target_speedup is None


def test_comparison_rejects_different_workloads() -> None:
    """Results with different workload identities cannot be compared."""
    reference = _result(
        workload_name="first",
        device="xpu",
        label="XPU BF16",
        actual_source="profiler",
        median_ms=2.0,
        p95_ms=2.0,
        operators=[],
    )
    target = _result(
        workload_name="second",
        device="cuda",
        label="A100 BF16",
        actual_source="profiler",
        median_ms=1.0,
        p95_ms=1.0,
        operators=[],
    )

    with pytest.raises(ComparisonError, match="workload mismatch"):
        compare_analyses(reference, target)
