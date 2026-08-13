# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: PLR2004, S101

"""Tests for profiler and unitrace operator attribution."""

import json
from pathlib import Path

import pytest
from perf_analysis import (
    analyze_collection,
    CollectionArtifacts,
    HardwareSpec,
    InvocationMetrics,
    LatencyStats,
    parse_unitrace_ops,
    TraceAnalysisError,
)


def _write_profiler(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "traceEvents": [
                    {
                        "ph": "X",
                        "cat": "cpu_op",
                        "name": "aten::mm",
                        "ts": 10,
                        "dur": 20,
                        "args": {
                            "External id": 7,
                            "Input Dims": "[[2,3],[3,4]]",
                            "Input Strides": "[]",
                        },
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


def test_unitrace_kernel_maps_by_order_name_and_external_id(tmp_path: Path) -> None:
    """A matching kernel maps its unitrace duration to the launching op."""
    profiler_path = tmp_path / "trace.json"
    unitrace_path = tmp_path / "python.1.json"
    _write_profiler(profiler_path)
    unitrace_path.write_text(
        json.dumps(
            {
                "traceEvents": [
                    {
                        "ph": "X",
                        "name": "gemm_kernel[SIMD16 {} {}]",
                        "ts": 12,
                        "dur": 6,
                    },
                ],
            },
        ),
        encoding="utf-8",
    )

    operations = parse_unitrace_ops(profiler_path, unitrace_path)

    assert operations[0].name == "aten::mm"
    assert operations[0].gpu_duration_us == 6
    assert operations[0].timestamp_us == 10
    assert operations[0].external_id == 7
    assert operations[0].source_index == 0


def test_unitrace_kernel_name_mismatch_is_skipped_and_reported(
    tmp_path: Path,
) -> None:
    """A mismatched kernel is omitted without discarding unitrace results."""
    profiler_path = tmp_path / "trace.json"
    unitrace_path = tmp_path / "python.1.json"
    _write_profiler(profiler_path)
    unitrace_path.write_text(
        json.dumps(
            {
                "traceEvents": [
                    {"ph": "X", "name": "wrong_kernel", "ts": 12, "dur": 6},
                ],
            },
        ),
        encoding="utf-8",
    )

    diagnostics: list[str] = []
    operations = parse_unitrace_ops(
        profiler_path,
        unitrace_path,
        diagnostics=diagnostics,
    )

    assert operations[0].gpu_duration_us == 0
    assert diagnostics == [
        (
            "unitrace kernel skipped at index 0: "
            "profiler='gemm_kernel', unitrace='wrong_kernel'"
        ),
    ]


def test_kernel_name_mismatch_does_not_trigger_profiler_fallback(
    tmp_path: Path,
) -> None:
    """Analysis keeps unitrace as its source and reports skipped kernels."""
    profiler_path = tmp_path / "trace.json"
    unitrace_path = tmp_path / "python.1.json"
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
                    {
                        "ph": "X",
                        "cat": "cpu_op",
                        "name": "aten::relu",
                        "ts": 40,
                        "dur": 20,
                        "args": {"External id": 8},
                    },
                    {
                        "ph": "X",
                        "cat": "kernel",
                        "name": "relu_kernel",
                        "ts": 42,
                        "dur": 8,
                        "args": {"External id": 8},
                    },
                ],
            },
        ),
        encoding="utf-8",
    )
    unitrace_path.write_text(
        json.dumps(
            {
                "traceEvents": [
                    {"ph": "X", "name": "wrong_kernel", "ts": 12, "dur": 6},
                    {"ph": "X", "name": "relu_kernel", "ts": 42, "dur": 4},
                ],
            },
        ),
        encoding="utf-8",
    )
    collection = CollectionArtifacts(
        schema_version=1,
        workload_name="mismatch",
        pid=1,
        device="xpu",
        hardware=HardwareSpec("synthetic", 1.0, 1.0),
        latency=LatencyStats([1.0], 1.0, 1.0, 1.0, 1.0),
        invocations=[
            InvocationMetrics("aten::mm", 1, 0, 1, 1),
            InvocationMetrics("aten::relu", 1, 0, 1, 1),
        ],
        totals={"flops": 2, "memory_bytes": 2},
        unaccounted_flop_ops={},
        trace_path=profiler_path,
        unitrace_glob=str(unitrace_path),
    )

    result = analyze_collection(collection)

    assert result.actual_source == "unitrace"
    assert result.t2_device_ms == pytest.approx(0.004)
    assert result.diagnostics == [
        (
            "unitrace kernel skipped at index 0: "
            "profiler='gemm_kernel', unitrace='wrong_kernel'"
        ),
    ]


def test_unitrace_kernel_count_mismatch_remains_an_error(tmp_path: Path) -> None:
    """A different number of kernels remains a structural mapping failure."""
    profiler_path = tmp_path / "trace.json"
    unitrace_path = tmp_path / "python.1.json"
    _write_profiler(profiler_path)
    unitrace_path.write_text(json.dumps({"traceEvents": []}), encoding="utf-8")

    with pytest.raises(TraceAnalysisError, match="kernel count mismatch"):
        parse_unitrace_ops(profiler_path, unitrace_path)
