# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: S101

"""Tests for structured and human-readable reports."""

import json
from pathlib import Path

import pytest

from test.perf_analysis import (
    AnalysisResult,
    compare_analyses,
    HardwareSpec,
    LatencyStats,
    OperatorAnalysis,
    render_comparison_markdown,
    render_comparison_text,
    render_markdown,
    render_text,
    write_comparison_json,
    write_json,
)


def _result() -> AnalysisResult:
    return AnalysisResult(
        schema_version=1,
        workload_name="tiny",
        device="xpu",
        hardware=HardwareSpec("test fp16", 10.0, 100.0),
        latency=LatencyStats([2.0], 2.0, 2.0, 2.0, 2.0),
        t1_ms=1.0,
        t2_wall_ms=2.0,
        t2_device_ms=1.5,
        actual_source="unitrace",
        totals={"flops": 10, "memory_bytes": 20},
        operators=[
            OperatorAnalysis(
                name="aten::mm",
                projected_ms=1.0,
                actual_ms=1.5,
                flops=10,
                memory_bytes=20,
                projected_calls=1,
                actual_calls=1,
                compute_bound_calls=1,
                memory_bound_calls=0,
            ),
        ],
    )


def test_all_report_forms_include_efficiency(tmp_path: Path) -> None:
    """Dataclass, JSON, text, and Markdown expose the same efficiency."""
    result = _result()
    output_path = write_json(result, tmp_path / "analysis.json")
    data = json.loads(output_path.read_text(encoding="utf-8"))

    assert result.efficiency == pytest.approx(0.5)
    assert data["efficiency"] == pytest.approx(0.5)
    assert "R = T1/T2 wall: 0.5000" in render_text(result)
    assert "| R = T1 / T2 wall | 0.5000 |" in render_markdown(result)


def test_comparison_reports_share_speedup_direction_and_sources(
    tmp_path: Path,
) -> None:
    """All comparison forms report reference-over-target wall speedup."""
    reference = _result()
    target = AnalysisResult(
        schema_version=1,
        workload_name="tiny",
        device="cuda",
        hardware=HardwareSpec("A100 BF16", 20.0, 200.0),
        latency=LatencyStats([0.5], 0.5, 1.0, 0.5, 1.0),
        t1_ms=0.5,
        t2_wall_ms=0.5,
        t2_device_ms=0.4,
        actual_source="profiler",
        totals={"flops": 10, "memory_bytes": 20},
        operators=[
            OperatorAnalysis(
                name="aten::mm",
                projected_ms=0.5,
                actual_ms=0.4,
                flops=10,
                memory_bytes=20,
                projected_calls=1,
                actual_calls=1,
                compute_bound_calls=1,
                memory_bound_calls=0,
            ),
        ],
        diagnostics=["target diagnostic"],
    )
    comparison = compare_analyses(
        reference,
        target,
        reference_name="XPU",
        target_name="A100",
    )
    data = json.loads(
        write_comparison_json(
            comparison,
            tmp_path / "comparison.json",
        ).read_text(encoding="utf-8"),
    )
    text = render_comparison_text(comparison)
    markdown = render_comparison_markdown(comparison)

    assert comparison.wall_speedup == pytest.approx(4.0)
    assert data["wall_speedup"] == pytest.approx(4.0)
    assert "Target wall speedup: 4.0000x" in text
    assert "| Actual source | unitrace | profiler | n/a |" in markdown
    assert "4.0000x speedup" in markdown
    assert "target diagnostic" in markdown
