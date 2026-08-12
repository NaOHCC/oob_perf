# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: S101

"""Integration tests for callable artifact collection."""

from pathlib import Path

import torch

from test.perf_analysis import collect_callable, CollectionConfig, HardwareSpec


def test_cpu_callable_writes_profiler_and_manifest(tmp_path: Path) -> None:
    """Profiler-only collection runs without the benchmark repository."""
    left = torch.randn(8, 16)
    right = torch.randn(16, 4)

    collection = collect_callable(
        lambda: torch.relu(left @ right),
        workload_name="tiny-mm",
        hardware=HardwareSpec("CPU FP32", 1.0, 1.0),
        config=CollectionConfig(
            device="cpu",
            output_dir=tmp_path,
            warmup_iterations=1,
            measurement_iterations=2,
            require_unitrace=False,
        ),
    )

    assert collection.trace_path.is_file()
    assert (tmp_path / "collection.json").is_file()
    assert collection.actual_source_preference == "profiler"
    assert collection.latency.median_ms > 0
    assert collection.totals["flops"] > 0
