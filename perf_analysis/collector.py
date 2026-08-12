# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Collect local performance artifacts from a repeatable callable."""

from __future__ import annotations

import json
import os
import statistics
import time
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from perf_analysis.metrics import MetricCount
from perf_analysis.models import (
    CollectionArtifacts,
    CollectionConfig,
    HardwareSpec,
    InvocationMetrics,
    LatencyStats,
    SCHEMA_VERSION,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any


class CollectionError(RuntimeError):
    """Raised when the requested collection cannot produce valid artifacts."""


def _synchronize(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "xpu":
        torch.xpu.synchronize()


def _validate_device(device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        msg = "CUDA collection requested, but CUDA is not available"
        raise CollectionError(msg)
    if device == "xpu" and not torch.xpu.is_available():
        msg = "XPU collection requested, but XPU is not available"
        raise CollectionError(msg)


def _validate_unitrace(output_dir: Path) -> None:
    if os.environ.get("UNITRACE_StartPaused") != "1":  # noqa: SIM112
        msg = "unitrace collection requires an externally wrapped process with --start-paused"
        raise CollectionError(msg)
    trace_output_dir = os.environ.get("UNITRACE_TraceOutputDir")  # noqa: SIM112
    if trace_output_dir is None:
        msg = "unitrace collection requires --output-dir-path"
        raise CollectionError(msg)
    if Path(trace_output_dir).resolve() != output_dir.resolve():
        msg = f"unitrace output directory does not match CollectionConfig.output_dir: {trace_output_dir}"
        raise CollectionError(msg)


def _percentile(samples: list[float], percentile: float) -> float:
    ordered = sorted(samples)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _measure_latency(
    workload: Callable[[], Any],
    *,
    device: str,
    iterations: int,
) -> LatencyStats:
    samples_ms = []
    for _ in range(iterations):
        _synchronize(device)
        start_ns = time.perf_counter_ns()
        workload()
        _synchronize(device)
        samples_ms.append((time.perf_counter_ns() - start_ns) / 1e6)
    return LatencyStats(
        samples_ms=samples_ms,
        median_ms=statistics.median(samples_ms),
        p95_ms=_percentile(samples_ms, 0.95),
        minimum_ms=min(samples_ms),
        maximum_ms=max(samples_ms),
    )


def _profiler_activities(device: str) -> list[torch.profiler.ProfilerActivity]:
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    elif device == "xpu":
        activities.append(torch.profiler.ProfilerActivity.XPU)
    return activities


def _collect_trace(
    workload: Callable[[], Any],
    *,
    device: str,
    trace_path: Path,
    unitrace_enabled: bool,
) -> None:
    with torch.profiler.profile(
        activities=_profiler_activities(device),
        acc_events=True,
        record_shapes=True,
    ) as profiler:
        if unitrace_enabled:
            os.environ["PTI_ENABLE_COLLECTION"] = "1"
        try:
            _synchronize(device)
            workload()
            _synchronize(device)
        finally:
            if unitrace_enabled:
                os.environ["PTI_ENABLE_COLLECTION"] = "0"
    profiler.export_chrome_trace(str(trace_path))


def collect_callable(
    workload: Callable[[], Any],
    *,
    workload_name: str,
    hardware: HardwareSpec,
    config: CollectionConfig,
) -> CollectionArtifacts:
    """Collect timing, operation metrics, and aligned profiler artifacts.

    The callable runs repeatedly for warmup, timing, metric counting, and trace
    collection. It must preserve equivalent semantics across invocations.

    Args:
        workload: Repeatable zero-argument callable that runs the model.
        workload_name: Name stored in artifacts and reports.
        hardware: Explicit roofline limits for the workload precision.
        config: Device, output, and iteration settings.

    Returns:
        The collection manifest. Unitrace output becomes readable after the
        externally wrapped process exits.

    Raises:
        ValueError: If the workload name is empty.
    """
    if not workload_name.strip():
        msg = "workload_name must not be empty"
        raise ValueError(msg)
    _validate_device(config.device)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    if config.require_unitrace:
        _validate_unitrace(config.output_dir)

    for _ in range(config.warmup_iterations):
        workload()
    _synchronize(config.device)

    latency = _measure_latency(
        workload,
        device=config.device,
        iterations=config.measurement_iterations,
    )

    with MetricCount(record_invocations=True) as metrics:
        workload()
    _synchronize(config.device)

    trace_path = (config.output_dir / "trace.json").resolve()
    _collect_trace(
        workload,
        device=config.device,
        trace_path=trace_path,
        unitrace_enabled=config.require_unitrace,
    )

    manifest = CollectionArtifacts(
        schema_version=SCHEMA_VERSION,
        workload_name=workload_name,
        pid=os.getpid(),
        device=config.device,
        hardware=hardware,
        latency=latency,
        invocations=[
            InvocationMetrics(
                name=item.name,
                exact_flops=item.exact_flops,
                estimated_flops=item.estimated_flops,
                gemm_attention_flops=item.gemm_attention_flops,
                memory_bytes=item.memory_bytes,
            )
            for item in metrics.invocations
        ],
        totals=metrics.summary(),
        unaccounted_flop_ops=metrics.unaccounted_flop_ops,
        trace_path=trace_path,
        unitrace_glob=str(config.output_dir.resolve() / f"*.{os.getpid()}*.json"),
        actual_source_preference=(
            "unitrace" if config.require_unitrace else "profiler"
        ),
    )
    manifest_path = config.output_dir / "collection.json"
    manifest_path.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest
