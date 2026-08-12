# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Parse and align PyTorch profiler and unitrace Chrome traces."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path


class TraceAnalysisError(RuntimeError):
    """Raised when trace artifacts cannot be mapped without losing fidelity."""


@dataclass(frozen=True)
class ActualOp:
    """Measured device time attributed to one dispatched operator."""

    name: str
    raw_name: str
    gpu_duration_us: float
    input_dims: str
    input_strides: str


@dataclass
class _CpuOp:
    raw_name: str
    timestamp: float
    end: float
    external_id: int | None
    input_dims: str
    input_strides: str


@dataclass(frozen=True)
class _Kernel:
    name: str
    timestamp: float
    duration_us: float
    external_id: int | None = None


_SIMD_SUFFIX = re.compile(r"\[SIMD\d+\s+\{[^}]*\}\s+\{[^}]*\}\]$")
_IMPL_DETAIL_OPS = frozenset({"aten::copy_"})
_NESTED_IMPL_DETAIL_OPS = frozenset(
    {
        "aten::empty",
        "aten::empty_like",
        "aten::empty_strided",
        "aten::fill_",
        "aten::new_empty",
        "aten::new_zeros",
        "aten::resize_",
        "aten::zero_",
        "aten::zeros",
    },
)


def normalize_op_name(name: str) -> str:  # noqa: PLR0911
    """Normalize dispatch wrappers and backend variants to comparable names.

    Returns:
        The canonical operator name used to join projected and measured data.
    """
    convolution_names = {
        "aten::_convolution",
        "aten::conv2d",
        "aten::convolution",
        "aten::convolution_overrideable",
        "aten::cudnn_convolution",
        "aten::mkldnn_convolution",
        "aten::xpu_convolution",
    }
    if name in convolution_names:
        return "aten::convolution"
    convolution_backward_names = {
        "aten::convolution_backward",
        "aten::convolution_backward_overrideable",
        "aten::cudnn_convolution_backward",
        "aten::mkldnn_convolution_backward",
    }
    if name in convolution_backward_names:
        return "aten::convolution_backward"
    if name in {
        "aten::contiguous",
        "aten::reshape",
        "aten::t",
        "aten::unbind",
        "aten::unbind.int",
    }:
        return "__view_noop__"

    wrapper_names = {
        "aten::_batch_norm_impl_index": "aten::native_batch_norm",
        "aten::adaptive_avg_pool2d": "aten::mean",
        "aten::batch_norm": "aten::native_batch_norm",
        "aten::clamp_min": "aten::relu",
        "aten::clamp_min_": "aten::relu_",
        "aten::layer_norm": "aten::native_layer_norm",
        "aten::linear": "aten::addmm",
        "aten::log_softmax": "aten::_log_softmax",
        "aten::matmul": "aten::bmm",
        "aten::max_pool2d": "aten::max_pool2d_with_indices",
        "aten::nll_loss": "aten::nll_loss_forward",
        "aten::nll_loss_nd": "aten::nll_loss_forward",
        "aten::softmax": "aten::_softmax",
    }
    if name in wrapper_names:
        return wrapper_names[name]

    if name in {
        "aten::_efficient_attention_forward",
        "aten::_flash_attention_forward",
        "aten::_scaled_dot_product_efficient_attention",
        "aten::_scaled_dot_product_flash_attention",
        "aten::_scaled_dot_product_fused_attention_overrideable",
        "aten::scaled_dot_product_attention",
    }:
        return "aten::sdpa_forward"
    if name in {
        "aten::_efficient_attention_backward",
        "aten::_flash_attention_backward",
        "aten::_scaled_dot_product_flash_attention_backward",
        "aten::_scaled_dot_product_fused_attention_overrideable_backward",
    }:
        return "aten::sdpa_backward"

    if "." in name.rsplit("::", maxsplit=1)[-1]:
        name = name.rsplit(".", 1)[0]
    return {
        "aten::add_": "aten::add",
        "aten::div_": "aten::div",
        "aten::masked_fill_": "aten::masked_fill",
        "aten::mul_": "aten::mul",
        "aten::sub_": "aten::sub",
    }.get(name, name)


def _read_events(path: Path) -> list[dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        events = data["traceEvents"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        msg = f"invalid Chrome trace: {path}"
        raise TraceAnalysisError(msg) from error
    if not isinstance(events, list):
        msg = f"traceEvents must be a list: {path}"
        raise TraceAnalysisError(msg)
    return [event for event in events if isinstance(event, dict)]


def _load_profiler(path: Path) -> tuple[list[_CpuOp], list[_Kernel], list[_Kernel]]:
    cpu_ops = []
    kernels = []
    device_events = []
    for event in _read_events(path):
        if event.get("ph") != "X":
            continue
        category = event.get("cat")
        args = event.get("args", {})
        args = args if isinstance(args, dict) else {}
        external_id = args.get("External id")
        external_id = external_id if isinstance(external_id, int) else None
        if category == "cpu_op" and str(event.get("name", "")).startswith("aten::"):
            timestamp = float(event.get("ts", 0))
            duration = float(event.get("dur", 0))
            cpu_ops.append(
                _CpuOp(
                    raw_name=str(event.get("name", "")),
                    timestamp=timestamp,
                    end=timestamp + duration,
                    external_id=external_id,
                    input_dims=str(args.get("Input Dims", "")),
                    input_strides=str(args.get("Input Strides", "")),
                ),
            )
        elif category in {"kernel", "gpu_memcpy"}:
            kernel = _Kernel(
                name=str(event.get("name", "")),
                timestamp=float(event.get("ts", 0)),
                duration_us=float(event.get("dur", 0)),
                external_id=external_id,
            )
            device_events.append(kernel)
            if category == "kernel":
                kernels.append(kernel)
    cpu_ops.sort(key=lambda item: item.timestamp)
    kernels.sort(key=lambda item: item.timestamp)
    device_events.sort(key=lambda item: item.timestamp)
    return cpu_ops, kernels, device_events


def _load_unitrace(path: Path) -> list[_Kernel]:
    kernels = []
    for event in _read_events(path):
        name = str(event.get("name", ""))
        if event.get("ph") != "X" or "dur" not in event or name.startswith("ze"):
            continue
        kernels.append(
            _Kernel(
                name=name,
                timestamp=float(event.get("ts", 0)),
                duration_us=float(event["dur"]),
            ),
        )
    kernels.sort(key=lambda item: item.timestamp)
    return kernels


def _parent_and_impl_detail(cpu_ops: list[_CpuOp]) -> tuple[list[int], list[bool]]:
    parent_indices = [-1] * len(cpu_ops)
    stack: list[tuple[float, int]] = []
    for index, operation in enumerate(cpu_ops):
        while stack and stack[-1][0] <= operation.timestamp:
            stack.pop()
        if stack:
            parent_indices[index] = stack[-1][1]
        stack.append((operation.end, index))

    normalized = [normalize_op_name(operation.raw_name) for operation in cpu_ops]
    impl_detail = [False] * len(cpu_ops)
    for index, operation in enumerate(cpu_ops):
        parent_index = parent_indices[index]
        if operation.raw_name in _IMPL_DETAIL_OPS or (
            parent_index >= 0
            and (
                normalized[index] == normalized[parent_index]
                or operation.raw_name in _NESTED_IMPL_DETAIL_OPS
            )
        ):
            impl_detail[index] = True
    return parent_indices, impl_detail


def _attribute_times(
    cpu_ops: list[_CpuOp],
    durations_by_external_id: dict[int, float],
) -> list[ActualOp]:
    parent_indices, impl_detail = _parent_and_impl_detail(cpu_ops)
    durations = [
        (
            durations_by_external_id.get(operation.external_id, 0.0)
            if operation.external_id is not None
            else 0.0
        )
        for operation in cpu_ops
    ]
    for index, is_detail in enumerate(impl_detail):
        if not is_detail or parent_indices[index] < 0:
            continue
        target = parent_indices[index]
        while impl_detail[target] and parent_indices[target] >= 0:
            target = parent_indices[target]
        durations[target] += durations[index]

    return [
        ActualOp(
            name=normalize_op_name(operation.raw_name),
            raw_name=operation.raw_name,
            gpu_duration_us=durations[index],
            input_dims=operation.input_dims,
            input_strides=operation.input_strides,
        )
        for index, operation in enumerate(cpu_ops)
        if not impl_detail[index]
    ]


def parse_profiler_ops(path: Path) -> list[ActualOp]:
    """Return device durations from a PyTorch profiler Chrome trace.

    Returns:
        Measured operations with profiler device durations.
    """
    cpu_ops, _kernels, device_events = _load_profiler(path)
    durations: dict[int, float] = {}
    for event in device_events:
        if event.external_id is not None:
            durations[event.external_id] = (
                durations.get(event.external_id, 0.0) + event.duration_us
            )
    return _attribute_times(cpu_ops, durations)


def parse_unitrace_ops(
    profiler_path: Path,
    unitrace_path: Path,
    *,
    diagnostics: list[str] | None = None,
) -> list[ActualOp]:
    """Map unitrace kernel durations to profiler operator launches.

    Kernel-name mismatches are skipped because the remaining kernel pairs can
    still provide useful unitrace attribution. Details are appended to
    ``diagnostics`` when supplied.

    Returns:
        Measured operations with unitrace kernel durations.

    Raises:
        TraceAnalysisError: If kernel counts or External ids mismatch.
    """
    cpu_ops, profiler_kernels, _device_events = _load_profiler(profiler_path)
    unitrace_kernels = _load_unitrace(unitrace_path)
    if len(profiler_kernels) != len(unitrace_kernels):
        msg = f"kernel count mismatch: profiler has {len(profiler_kernels)}, unitrace has {len(unitrace_kernels)}"
        raise TraceAnalysisError(msg)

    durations: dict[int, float] = {}
    for index, (profiler_kernel, unitrace_kernel) in enumerate(
        zip(profiler_kernels, unitrace_kernels, strict=True),
    ):
        unitrace_name = _SIMD_SUFFIX.sub("", unitrace_kernel.name).strip()
        if profiler_kernel.name != unitrace_name:
            if diagnostics is not None:
                diagnostics.append(
                    "unitrace kernel skipped at index "
                    f"{index}: profiler={profiler_kernel.name!r}, "
                    f"unitrace={unitrace_name!r}",
                )
            continue
        if profiler_kernel.external_id is None:
            msg = f"profiler kernel has no External id at index {index}"
            raise TraceAnalysisError(msg)
        durations[profiler_kernel.external_id] = (
            durations.get(profiler_kernel.external_id, 0.0)
            + unitrace_kernel.duration_us
        )
    return _attribute_times(cpu_ops, durations)
