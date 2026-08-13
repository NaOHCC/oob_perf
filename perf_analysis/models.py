# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Structured inputs and results for callable performance analysis."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Literal

SCHEMA_VERSION = 1
CALLS_SCHEMA_VERSION = 1
ActualSource = Literal["unitrace", "profiler"]
CallMatchStatus = Literal["sequence-paired", "actual-only", "projected-only"]
CallBound = Literal["compute", "memory", "none"]


@dataclass(frozen=True)
class HardwareSpec:
    """The hardware limits used by the roofline projection.

    Args:
        label: Human-readable hardware and precision label.
        peak_tflops: Peak throughput for the workload precision in TFLOPS.
        memory_bandwidth_gbs: Peak DRAM bandwidth in GB/s.
    """

    label: str
    peak_tflops: float
    memory_bandwidth_gbs: float

    def __post_init__(self) -> None:
        """Validate positive hardware limits.

        Raises:
            ValueError: If the label is empty or either limit is not positive.
        """
        if not self.label.strip():
            msg = "hardware label must not be empty"
            raise ValueError(msg)
        if self.peak_tflops <= 0:
            msg = "peak_tflops must be greater than zero"
            raise ValueError(msg)
        if self.memory_bandwidth_gbs <= 0:
            msg = "memory_bandwidth_gbs must be greater than zero"
            raise ValueError(msg)

    @property
    def peak_flops_per_second(self) -> float:
        """Peak throughput in FLOPs per second."""
        return self.peak_tflops * 1e12

    @property
    def memory_bytes_per_second(self) -> float:
        """Peak DRAM bandwidth in bytes per second."""
        return self.memory_bandwidth_gbs * 1e9


@dataclass(frozen=True)
class CollectionConfig:
    """Controls repeated callable execution and trace collection."""

    device: str
    output_dir: Path
    warmup_iterations: int = 5
    measurement_iterations: int = 20
    require_unitrace: bool = True

    def __post_init__(self) -> None:
        """Validate collection settings and normalize the output path.

        Raises:
            ValueError: If the device or iteration counts are invalid.
        """
        if self.device not in {"cpu", "cuda", "xpu"}:
            msg = f"unsupported device: {self.device}"
            raise ValueError(msg)
        if self.warmup_iterations < 0:
            msg = "warmup_iterations must not be negative"
            raise ValueError(msg)
        if self.measurement_iterations <= 0:
            msg = "measurement_iterations must be greater than zero"
            raise ValueError(msg)
        if self.require_unitrace and self.device != "xpu":
            msg = "unitrace collection is supported only for XPU"
            raise ValueError(msg)
        object.__setattr__(self, "output_dir", Path(self.output_dir))


@dataclass(frozen=True)
class LatencyStats:
    """Synchronized wall-clock latency samples and summary statistics."""

    samples_ms: list[float]
    median_ms: float
    p95_ms: float
    minimum_ms: float
    maximum_ms: float


@dataclass(frozen=True)
class InvocationMetrics:
    """Serializable metrics for one dispatched operator invocation."""

    name: str
    exact_flops: int
    estimated_flops: int
    gemm_attention_flops: int
    memory_bytes: int

    @property
    def flops(self) -> int:
        """Exact and estimated FLOPs combined."""
        return self.exact_flops + self.estimated_flops


@dataclass(frozen=True)
class CallRecord:
    """Projected and measured metrics for one operator-local call."""

    name: str
    match_status: CallMatchStatus
    projected_raw_name: str | None
    actual_raw_name: str | None
    projected_index: int | None
    actual_index: int | None
    external_id: int | None
    projected_ms: float | None
    compute_ms: float | None
    memory_ms: float | None
    flops: int | None
    memory_bytes: int | None
    bound: CallBound | None
    actual_ms: float | None
    timestamp_us: float | None
    input_dims: str | None
    input_strides: str | None

    @property
    def efficiency(self) -> float | None:
        """Per-call roofline efficiency for paired records."""
        if self.projected_ms is None or self.actual_ms is None or self.actual_ms <= 0:
            return None
        return self.projected_ms / self.actual_ms


@dataclass(frozen=True)
class CallDataset:
    """Versioned browser artifact containing all projected and measured calls."""

    schema_version: int
    workload_name: str
    device: str
    actual_source: ActualSource
    calls: list[CallRecord]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible dataset with derived efficiencies."""
        data = asdict(self)
        for call_data, call in zip(data["calls"], self.calls, strict=True):
            call_data["efficiency"] = call.efficiency
        return data


@dataclass(frozen=True)
class CollectionArtifacts:
    """Manifest produced before the unitrace-wrapped process exits."""

    schema_version: int
    workload_name: str
    pid: int
    device: str
    hardware: HardwareSpec
    latency: LatencyStats
    invocations: list[InvocationMetrics]
    totals: dict[str, int]
    unaccounted_flop_ops: dict[str, int]
    trace_path: Path
    unitrace_glob: str
    actual_source_preference: ActualSource | None = None

    _PATH_FIELDS: ClassVar[frozenset[str]] = frozenset({"trace_path"})

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible manifest."""
        data = asdict(self)
        for field_name in self._PATH_FIELDS:
            data[field_name] = str(data[field_name])
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CollectionArtifacts:
        """Build a manifest from JSON-compatible data.

        Returns:
            The deserialized collection.

        Raises:
            ValueError: If the schema version is unsupported.
        """
        if data.get("schema_version") != SCHEMA_VERSION:
            msg = f"unsupported collection schema: {data.get('schema_version')}"
            raise ValueError(msg)
        return cls(
            schema_version=data["schema_version"],
            workload_name=data["workload_name"],
            pid=data["pid"],
            device=data["device"],
            hardware=HardwareSpec(**data["hardware"]),
            latency=LatencyStats(**data["latency"]),
            invocations=[InvocationMetrics(**item) for item in data["invocations"]],
            totals=data["totals"],
            unaccounted_flop_ops=data["unaccounted_flop_ops"],
            trace_path=Path(data["trace_path"]),
            unitrace_glob=data["unitrace_glob"],
            actual_source_preference=data.get("actual_source_preference"),
        )


@dataclass(frozen=True)
class OperatorAnalysis:
    """Projected and measured metrics aggregated by canonical operator name."""

    name: str
    projected_ms: float
    actual_ms: float | None
    flops: int
    memory_bytes: int
    projected_calls: int
    actual_calls: int
    compute_bound_calls: int
    memory_bound_calls: int
    dominant_shape: str = ""
    dominant_stride: str = ""

    @property
    def efficiency(self) -> float | None:
        """Per-operator roofline efficiency when actual time exists."""
        if self.actual_ms is None or self.actual_ms <= 0:
            return None
        return self.projected_ms / self.actual_ms

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OperatorAnalysis:
        """Build an operator result while ignoring derived JSON fields.

        Returns:
            The deserialized operator result.
        """
        return cls(
            name=data["name"],
            projected_ms=data["projected_ms"],
            actual_ms=data["actual_ms"],
            flops=data["flops"],
            memory_bytes=data["memory_bytes"],
            projected_calls=data["projected_calls"],
            actual_calls=data["actual_calls"],
            compute_bound_calls=data["compute_bound_calls"],
            memory_bound_calls=data["memory_bound_calls"],
            dominant_shape=data.get("dominant_shape", ""),
            dominant_stride=data.get("dominant_stride", ""),
        )


@dataclass(frozen=True)
class AnalysisResult:
    """Complete single-workload roofline analysis."""

    schema_version: int
    workload_name: str
    device: str
    hardware: HardwareSpec
    latency: LatencyStats
    t1_ms: float
    t2_wall_ms: float
    t2_device_ms: float | None
    actual_source: ActualSource
    totals: dict[str, int]
    operators: list[OperatorAnalysis]
    diagnostics: list[str] = field(default_factory=list)

    @property
    def efficiency(self) -> float:
        """End-to-end roofline efficiency using wall-clock T2."""
        return self.t1_ms / self.t2_wall_ms

    @property
    def device_efficiency(self) -> float | None:
        """Diagnostic efficiency using summed device time."""
        if self.t2_device_ms is None or self.t2_device_ms <= 0:
            return None
        return self.t1_ms / self.t2_device_ms

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible analysis result."""
        data = asdict(self)
        data["efficiency"] = self.efficiency
        data["device_efficiency"] = self.device_efficiency
        for operator_data, operator in zip(
            data["operators"],
            self.operators,
            strict=True,
        ):
            operator_data["efficiency"] = operator.efficiency
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AnalysisResult:
        """Build an analysis result while recomputing derived metrics.

        Returns:
            The deserialized analysis result.

        Raises:
            ValueError: If the schema version or actual source is unsupported.
        """
        if data.get("schema_version") != SCHEMA_VERSION:
            msg = f"unsupported analysis schema: {data.get('schema_version')}"
            raise ValueError(msg)
        actual_source = data["actual_source"]
        if actual_source not in {"unitrace", "profiler"}:
            msg = f"unsupported actual source: {actual_source}"
            raise ValueError(msg)
        return cls(
            schema_version=data["schema_version"],
            workload_name=data["workload_name"],
            device=data["device"],
            hardware=HardwareSpec(**data["hardware"]),
            latency=LatencyStats(**data["latency"]),
            t1_ms=data["t1_ms"],
            t2_wall_ms=data["t2_wall_ms"],
            t2_device_ms=data["t2_device_ms"],
            actual_source=actual_source,
            totals=data["totals"],
            operators=[OperatorAnalysis.from_dict(item) for item in data["operators"]],
            diagnostics=data.get("diagnostics", []),
        )


@dataclass(frozen=True)
class OperatorComparison:
    """One canonical operator compared between reference and target devices."""

    name: str
    reference_actual_ms: float | None
    target_actual_ms: float | None
    target_speedup: float | None
    reference_minus_target_ms: float | None
    reference_projected_ms: float | None
    target_projected_ms: float | None
    reference_efficiency: float | None
    target_efficiency: float | None
    reference_calls: int | None
    target_calls: int | None
    reference_bound: str | None
    target_bound: str | None


@dataclass(frozen=True)
class ComparisonResult:
    """Offline comparison of one workload on a reference and target device."""

    schema_version: int
    workload_name: str
    reference_name: str
    target_name: str
    reference: AnalysisResult
    target: AnalysisResult
    wall_speedup: float
    p95_speedup: float
    operators: list[OperatorComparison]
    comparable: bool
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible comparison result."""
        return {
            "schema_version": self.schema_version,
            "workload_name": self.workload_name,
            "reference_name": self.reference_name,
            "target_name": self.target_name,
            "reference": self.reference.to_dict(),
            "target": self.target.to_dict(),
            "wall_speedup": self.wall_speedup,
            "p95_speedup": self.p95_speedup,
            "operators": [asdict(operator) for operator in self.operators],
            "comparable": self.comparable,
            "warnings": self.warnings,
        }
