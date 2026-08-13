# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Analyze one repeatable callable without the benchmark repository."""

from perf_analysis.analyzer import (
    analyze_collection,
    analyze_collection_artifacts,
    load_collection,
)
from perf_analysis.collector import collect_callable, CollectionError
from perf_analysis.comparison import (
    compare_analyses,
    ComparisonError,
    load_analysis,
    load_reference,
)
from perf_analysis.metrics import MetricCount, OpInvocation, OpMetrics
from perf_analysis.models import (
    AnalysisResult,
    CallDataset,
    CallRecord,
    CollectionArtifacts,
    CollectionConfig,
    ComparisonResult,
    HardwareSpec,
    InvocationMetrics,
    LatencyStats,
    OperatorAnalysis,
    OperatorComparison,
)
from perf_analysis.reporting import (
    render_comparison_markdown,
    render_comparison_text,
    render_markdown,
    render_text,
    write_calls_json,
    write_comparison_json,
    write_comparison_markdown,
    write_json,
    write_markdown,
)
from perf_analysis.traces import (
    ActualOp,
    normalize_op_name,
    parse_profiler_ops,
    parse_unitrace_ops,
    TraceAnalysisError,
)

__all__ = [
    "ActualOp",
    "AnalysisResult",
    "CallDataset",
    "CallRecord",
    "CollectionArtifacts",
    "CollectionConfig",
    "CollectionError",
    "ComparisonError",
    "ComparisonResult",
    "HardwareSpec",
    "InvocationMetrics",
    "LatencyStats",
    "MetricCount",
    "OpInvocation",
    "OpMetrics",
    "OperatorAnalysis",
    "OperatorComparison",
    "TraceAnalysisError",
    "analyze_collection",
    "analyze_collection_artifacts",
    "collect_callable",
    "compare_analyses",
    "load_analysis",
    "load_collection",
    "load_reference",
    "normalize_op_name",
    "parse_profiler_ops",
    "parse_unitrace_ops",
    "render_comparison_markdown",
    "render_comparison_text",
    "render_markdown",
    "render_text",
    "write_calls_json",
    "write_comparison_json",
    "write_comparison_markdown",
    "write_json",
    "write_markdown",
]
