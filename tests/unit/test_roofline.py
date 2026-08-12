# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: S101

"""Tests for metric collection and per-invocation roofline projection."""

import json
from pathlib import Path

import pytest
import torch

from test.perf_analysis import (
    CollectionArtifacts,
    CollectionConfig,
    HardwareSpec,
    InvocationMetrics,
    LatencyStats,
    MetricCount,
    TraceAnalysisError,
    analyze_collection,
)


def test_invocation_metrics_preserve_aggregates() -> None:
    """Invocation records sum to the existing aggregate counters."""
    left = torch.randn(2, 3)
    right = torch.randn(3, 4)

    with MetricCount(record_invocations=True) as metrics:
        torch.relu(left @ right)

    assert [item.name for item in metrics.invocations] == ["aten::mm", "aten::relu"]
    assert sum(item.flops for item in metrics.invocations) == metrics.flops
    assert (
        sum(item.memory_bytes for item in metrics.invocations) == metrics.memory_bytes
    )


def test_projection_sums_per_invocation_bounds(tmp_path: Path) -> None:
    """Mixed bounds use sum(max(each call)) instead of max(aggregate sums)."""
    trace_path = tmp_path / "trace.json"
    trace_path.write_text(json.dumps({"traceEvents": []}), encoding="utf-8")
    collection = CollectionArtifacts(
        schema_version=1,
        workload_name="mixed-bound",
        pid=1,
        device="cpu",
        hardware=HardwareSpec("synthetic", 1.0, 1.0),
        latency=LatencyStats([3.0], 3.0, 3.0, 3.0, 3.0),
        invocations=[
            InvocationMetrics("aten::custom", 1_000_000_000, 0, 0, 1),
            InvocationMetrics("aten::custom", 1, 0, 0, 1_000_000),
        ],
        totals={"flops": 1_000_000_001, "memory_bytes": 1_000_001},
        unaccounted_flop_ops={},
        trace_path=trace_path,
        unitrace_glob=str(tmp_path / "missing.*.json"),
    )

    result = analyze_collection(collection, allow_profiler_fallback=True)

    assert result.t1_ms == pytest.approx(2.0)
    assert result.efficiency == pytest.approx(2.0 / 3.0)
    assert result.actual_source == "profiler"


def test_unitrace_is_strict_by_default(tmp_path: Path) -> None:
    """A missing unitrace artifact fails unless fallback is explicit."""
    trace_path = tmp_path / "trace.json"
    trace_path.write_text(json.dumps({"traceEvents": []}), encoding="utf-8")
    collection = CollectionArtifacts(
        schema_version=1,
        workload_name="strict",
        pid=1,
        device="xpu",
        hardware=HardwareSpec("synthetic", 1.0, 1.0),
        latency=LatencyStats([1.0], 1.0, 1.0, 1.0, 1.0),
        invocations=[],
        totals={},
        unaccounted_flop_ops={},
        trace_path=trace_path,
        unitrace_glob=str(tmp_path / "missing.*.json"),
    )

    with pytest.raises(TraceAnalysisError, match="expected one unitrace"):
        analyze_collection(collection)


@pytest.mark.parametrize("device", ["cpu", "cuda", "xpu"])
def test_profiler_preference_does_not_require_unitrace(
    tmp_path: Path,
    device: str,
) -> None:
    """An explicit profiler collection is native, not a unitrace fallback."""
    trace_path = tmp_path / "trace.json"
    trace_path.write_text(json.dumps({"traceEvents": []}), encoding="utf-8")
    collection = CollectionArtifacts(
        schema_version=1,
        workload_name="profiler-only",
        pid=1,
        device=device,
        hardware=HardwareSpec("synthetic", 1.0, 1.0),
        latency=LatencyStats([1.0], 1.0, 1.0, 1.0, 1.0),
        invocations=[],
        totals={},
        unaccounted_flop_ops={},
        trace_path=trace_path,
        unitrace_glob=str(tmp_path / "missing.*.json"),
        actual_source_preference="profiler",
    )

    result = analyze_collection(collection)

    assert result.actual_source == "profiler"
    assert not any("fallback" in diagnostic for diagnostic in result.diagnostics)


def test_collection_manifest_preserves_source_preference(tmp_path: Path) -> None:
    """New manifests persist source intent while old manifests remain readable."""
    data = {
        "schema_version": 1,
        "workload_name": "source",
        "pid": 1,
        "device": "xpu",
        "hardware": {
            "label": "synthetic",
            "peak_tflops": 1.0,
            "memory_bandwidth_gbs": 1.0,
        },
        "latency": {
            "samples_ms": [1.0],
            "median_ms": 1.0,
            "p95_ms": 1.0,
            "minimum_ms": 1.0,
            "maximum_ms": 1.0,
        },
        "invocations": [],
        "totals": {},
        "unaccounted_flop_ops": {},
        "trace_path": str(tmp_path / "trace.json"),
        "unitrace_glob": str(tmp_path / "*.json"),
    }

    old_collection = CollectionArtifacts.from_dict(data)
    data["actual_source_preference"] = "profiler"
    profiler_collection = CollectionArtifacts.from_dict(data)

    assert old_collection.actual_source_preference is None
    assert profiler_collection.actual_source_preference == "profiler"
    assert profiler_collection.to_dict()["actual_source_preference"] == "profiler"


def test_unitrace_configuration_requires_xpu(tmp_path: Path) -> None:
    """Unitrace collection rejects devices that do not use Level Zero."""
    with pytest.raises(ValueError, match="only for XPU"):
        CollectionConfig(device="cpu", output_dir=tmp_path)
