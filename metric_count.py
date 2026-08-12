# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: ANN401, DOC502, ERA001, PLR2004

"""Count MolmoAct2 inference FLOPs and logical ATen memory movement.

``MetricCount`` measures the tensors consumed and produced by each dispatched
ATen operation. Its memory result is a logical I/O volume, not a hardware
profiler's DRAM transaction count: it does not model cache residency, kernel
fusion beyond the dispatched operation, or allocator traffic.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

import humanize
import torch
from torch.utils._python_dispatch import TorchDispatchMode  # noqa: PLC2701

DEFAULT_DEVICE = "xpu"
DEFAULT_COMPILE = False


@dataclass
class OpMetrics:
    """Aggregate metrics for one dispatched ATen operator name."""

    calls: int = 0
    exact_flops: int = 0
    estimated_flops: int = 0
    gemm_attention_flops: int = 0
    memory_bytes: int = 0

    @property
    def flops(self) -> int:
        """Combined exact and estimated FLOPs."""
        return self.exact_flops + self.estimated_flops


@dataclass(frozen=True)
class OpInvocation:
    """Metrics for one dispatched ATen operator invocation."""

    name: str
    exact_flops: int
    estimated_flops: int
    gemm_attention_flops: int
    memory_bytes: int

    @property
    def flops(self) -> int:
        """Combined exact and estimated FLOPs."""
        return self.exact_flops + self.estimated_flops


class MetricCount(TorchDispatchMode):
    """Count FLOPs and logical tensor I/O for eager PyTorch execution.

    Matrix multiplication and the two attention matrix products use exact
    $2MNK$ FLOP accounting. Basic elementwise arithmetic is also exact under
    the convention that one scalar arithmetic operation is one FLOP.
    Transcendental functions, normalization, GELU, SiLU, and fused attention
    softmax use fixed pointwise-equivalent estimates, exposed separately in
    :attr:`estimated_flops`. Operations with floating outputs that have no
    accounting rule are reported through :attr:`unaccounted_flop_ops`.

    Memory movement counts logical tensor reads and writes per materialized
    ATen operation. View-like aliases count neither reads nor writes; indexed
    reads such as embedding and gather count only selected source elements.
    """

    _ALIAS_OPS: ClassVar[frozenset[str]] = frozenset(
        {
            "aten::alias",
            "aten::as_strided",
            "aten::detach",
            "aten::expand",
            "aten::lift_fresh",
            "aten::permute",
            "aten::select",
            "aten::slice",
            "aten::split",
            "aten::split_with_sizes",
            "aten::squeeze",
            "aten::t",
            "aten::transpose",
            "aten::unbind",
            "aten::unsqueeze",
            "aten::view",
            "aten::view_as",
            "aten::_local_scalar_dense",
            "aten::_reshape_alias",
            "aten::_unsafe_view",
        },
    )
    _DECOMPOSABLE_OPS: ClassVar[frozenset[str]] = frozenset(
        {"aten::linear", "aten::matmul"},
    )
    _NO_DATA_MOVE_CREATION_OPS: ClassVar[frozenset[str]] = frozenset(
        {
            "aten::empty",
            "aten::empty_like",
            "aten::empty_strided",
            "aten::new_empty",
            "aten::new_empty_strided",
        },
    )
    _OUTPUT_WRITE_CREATION_OPS: ClassVar[frozenset[str]] = frozenset(
        {
            "aten::arange",
            "aten::full",
            "aten::full_like",
            "aten::new_full",
            "aten::new_ones",
            "aten::new_zeros",
            "aten::ones",
            "aten::ones_like",
            "aten::scalar_tensor",
            "aten::zeros",
            "aten::zeros_like",
        },
    )
    _ZERO_FLOP_OPS: ClassVar[frozenset[str]] = frozenset(
        {
            "aten::arange",
            "aten::bitwise_and",
            "aten::bitwise_not",
            "aten::bitwise_or",
            "aten::cat",
            "aten::clone",
            "aten::copy_",
            "aten::embedding",
            "aten::eq",
            "aten::full",
            "aten::gather",
            "aten::ge",
            "aten::gt",
            "aten::index",
            "aten::index_put",
            "aten::index_select",
            "aten::le",
            "aten::lt",
            "aten::masked_fill",
            "aten::masked_select",
            "aten::ne",
            "aten::ones",
            "aten::rand",
            "aten::randn",
            "aten::scalar_tensor",
            "aten::scatter",
            "aten::take",
            "aten::tril",
            "aten::triu",
            "aten::_to_copy",
            "aten::zeros",
        },
    )

    def __init__(self, *, record_invocations: bool = False) -> None:
        """Initialize empty aggregate and per-operator counters.

        Args:
            record_invocations: Preserve metrics for every dispatched operation
                in execution order.
        """
        super().__init__()
        self.record_invocations = record_invocations
        self.exact_flops = 0
        self.estimated_flops = 0
        self.gemm_attention_flops = 0
        self.memory_bytes = 0
        self.op_metrics: dict[str, OpMetrics] = {}
        self.invocations: list[OpInvocation] = []
        self.unaccounted_flop_ops: dict[str, int] = {}

    @property
    def flops(self) -> int:
        """Combined exact and estimated FLOPs."""
        return self.exact_flops + self.estimated_flops

    @property
    def memory(self) -> int:
        """Logical memory movement in bytes."""
        return self.memory_bytes

    @property
    def memory_move_bytes(self) -> int:
        """Logical memory movement in bytes."""
        return self.memory_bytes

    def __torch_dispatch__(  # noqa: PLW3201
        self,
        func: Any,
        types: Any,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        """Execute an ATen operation and record its aggregate metrics.

        Returns:
            The operation output.
        """
        del types
        call_kwargs = {} if kwargs is None else kwargs
        op_name = func.name()

        if self._matches_any(op_name, self._DECOMPOSABLE_OPS):
            with self:
                decomposed_output = func.decompose(*args, **call_kwargs)
            if decomposed_output is not NotImplemented:
                return decomposed_output

        output = func(*args, **call_kwargs)
        is_alias = self._is_alias_operation(op_name, args, output)
        exact_flops, estimated_flops, gemm_attention_flops = self._flops_for_op(
            op_name,
            args,
            call_kwargs,
            output,
        )
        memory_bytes = (
            0
            if is_alias
            else self._memory_bytes_for_op(op_name, args, call_kwargs, output)
        )
        self._record(
            op_name,
            exact_flops,
            estimated_flops,
            gemm_attention_flops,
            memory_bytes,
        )

        if self._should_report_unaccounted_flops(
            op_name,
            output,
            is_alias=is_alias,
            exact_flops=exact_flops,
            estimated_flops=estimated_flops,
        ):
            self.unaccounted_flop_ops[op_name] = (
                self.unaccounted_flop_ops.get(op_name, 0) + 1
            )
        return output

    def summary(self) -> dict[str, int]:
        """Return aggregate counter values suitable for machine-readable output.

        Returns:
            Total FLOPs and logical memory movement by metric category.
        """
        return {
            "flops": self.flops,
            "exact_flops": self.exact_flops,
            "estimated_flops": self.estimated_flops,
            "gemm_attention_flops": self.gemm_attention_flops,
            "memory_bytes": self.memory_bytes,
        }

    def top_operations(self, limit: int | None = None) -> list[tuple[str, OpMetrics]]:
        """Return operations ordered by FLOPs and then logical memory movement.

        Returns:
            Operator names paired with their aggregate metrics.
        """
        operations = sorted(
            self.op_metrics.items(),
            key=lambda item: (item[1].flops, item[1].memory_bytes, item[1].calls),
            reverse=True,
        )
        return operations if limit is None else operations[:limit]

    def format_report(self, max_operations: int = 20) -> str:
        """Format aggregate and per-operator metrics for terminal output.

        Returns:
            A multi-line human-readable metrics report.
        """
        lines = [
            "MetricCount",
            f"  FLOPs: {self._format_flops(self.flops)}",
            f"    exact: {self._format_flops(self.exact_flops)}",
            f"    estimated: {self._format_flops(self.estimated_flops)}",
            f"    GEMM/attention: {self._format_flops(self.gemm_attention_flops)}",
            f"  Logical memory move volume: {self._format_bytes(self.memory_bytes)}",
        ]
        operations = self.top_operations(max_operations)
        if operations:
            lines.append(f"  Top {len(operations)} operations:")
            for op_name, metrics in operations:
                lines.append(
                    f"    {op_name}: calls={humanize.intcomma(metrics.calls)}, "
                    f"flops={self._format_flops(metrics.flops)}, "
                    f"gemm_attention_flops={self._format_flops(metrics.gemm_attention_flops)}, "
                    f"logical_bytes={self._format_bytes(metrics.memory_bytes)}",
                )
        if self.unaccounted_flop_ops:
            unknown = ", ".join(
                f"{op_name} ({humanize.intcomma(calls)})"
                for op_name, calls in sorted(self.unaccounted_flop_ops.items())
            )
            lines.append(f"  Unaccounted floating-output ops: {unknown}")
        return "\n".join(lines)

    def _record(
        self,
        op_name: str,
        exact_flops: int,
        estimated_flops: int,
        gemm_attention_flops: int,
        memory_bytes: int,
    ) -> None:
        metrics = self.op_metrics.setdefault(op_name, OpMetrics())
        metrics.calls += 1
        metrics.exact_flops += exact_flops
        metrics.estimated_flops += estimated_flops
        metrics.gemm_attention_flops += gemm_attention_flops
        metrics.memory_bytes += memory_bytes
        self.exact_flops += exact_flops
        self.estimated_flops += estimated_flops
        self.gemm_attention_flops += gemm_attention_flops
        self.memory_bytes += memory_bytes
        if self.record_invocations:
            self.invocations.append(
                OpInvocation(
                    name=op_name,
                    exact_flops=exact_flops,
                    estimated_flops=estimated_flops,
                    gemm_attention_flops=gemm_attention_flops,
                    memory_bytes=memory_bytes,
                ),
            )

    def _flops_for_op(  # noqa: C901, PLR0911, PLR0912
        self,
        op_name: str,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
        output: Any,
    ) -> tuple[int, int, int]:
        if not self._contains_floating_tensor(output):
            return 0, 0, 0

        if self._matches_op(op_name, "aten::addmm"):
            return self._addmm_flops(args, kwargs, output)
        if self._matches_op(op_name, "aten::mm"):
            return self._mm_flops(args)
        if self._matches_op(op_name, "aten::bmm"):
            return self._bmm_flops(args)
        if self._matches_op(op_name, "aten::matmul"):
            return self._matmul_flops(args, output)
        if self._matches_op(op_name, "aten::linear"):
            return self._linear_flops(args, output)
        if self._matches_op(op_name, "aten::_grouped_mm"):
            return self._grouped_mm_flops(args)
        if self._matches_op(op_name, "aten::_scaled_mm"):
            return self._scaled_mm_flops(args, output)
        if "scaled_dot_product" in op_name and "backward" not in op_name:
            return self._attention_flops(op_name, args, kwargs)

        output_elements = self._floating_output_elements(output)
        if output_elements == 0:
            return 0, 0, 0
        if self._matches_any(
            op_name,
            {
                "aten::add",
                "aten::add_",
                "aten::mul",
                "aten::mul_",
                "aten::sub",
                "aten::sub_",
                "aten::div",
                "aten::div_",
                "aten::neg",
                "aten::reciprocal",
            },
        ):
            return output_elements, 0, 0
        if self._matches_op(op_name, "aten::rsub"):
            return output_elements, 0, 0
        if self._matches_op(op_name, "aten::sum"):
            input_tensor = self._first_tensor(args)
            reductions = (
                0
                if input_tensor is None
                else max(input_tensor.numel() - output_elements, 0)
            )
            return 0, reductions, 0
        if self._matches_op(op_name, "aten::mean"):
            input_tensor = self._first_tensor(args)
            if input_tensor is None:
                return 0, 0, 0
            reductions = max(input_tensor.numel() - output_elements, 0)
            return 0, reductions + output_elements, 0
        if self._matches_op(op_name, "aten::native_layer_norm"):
            input_tensor = self._first_tensor(args)
            return 0, 0 if input_tensor is None else 5 * input_tensor.numel(), 0
        if self._matches_any(
            op_name,
            {"aten::_softmax", "aten::softmax", "aten::log_softmax"},
        ):
            return 0, 5 * output_elements, 0
        if self._matches_any(op_name, {"aten::silu", "aten::sigmoid", "aten::tanh"}):
            return 0, 4 * output_elements, 0
        if self._matches_op(op_name, "aten::gelu"):
            return 0, 8 * output_elements, 0
        if self._matches_any(
            op_name,
            {
                "aten::pow",
                "aten::rsqrt",
                "aten::exp",
                "aten::sin",
                "aten::cos",
                "aten::sqrt",
                "aten::clamp_min",
                "aten::relu",
                "aten::where",
            },
        ):
            return 0, output_elements, 0
        return 0, 0, 0

    def _addmm_flops(
        self,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
        output: Any,
    ) -> tuple[int, int, int]:
        if (
            len(args) < 3
            or not isinstance(args[1], torch.Tensor)
            or not isinstance(args[2], torch.Tensor)
        ):
            return 0, 0, 0
        mat1, mat2 = args[1], args[2]
        if mat1.dim() != 2 or mat2.dim() != 2:
            return 0, 0, 0
        output_elements = self._floating_output_elements(output)
        matrix_flops = 2 * mat1.shape[0] * mat1.shape[1] * mat2.shape[1]
        alpha = self._scalar_argument(args, kwargs, 4, "alpha", 1)
        beta = self._scalar_argument(args, kwargs, 3, "beta", 1)
        scalar_flops = 0
        if alpha != 1:
            scalar_flops += output_elements
        if beta != 0:
            scalar_flops += output_elements
            if beta != 1:
                scalar_flops += output_elements
        exact_flops = matrix_flops + scalar_flops
        return exact_flops, 0, matrix_flops

    @staticmethod
    def _mm_flops(args: tuple[Any, ...]) -> tuple[int, int, int]:
        if (
            len(args) < 2
            or not isinstance(args[0], torch.Tensor)
            or not isinstance(args[1], torch.Tensor)
        ):
            return 0, 0, 0
        mat1, mat2 = args[0], args[1]
        if mat1.dim() != 2 or mat2.dim() != 2:
            return 0, 0, 0
        matrix_flops = 2 * mat1.shape[0] * mat1.shape[1] * mat2.shape[1]
        return matrix_flops, 0, matrix_flops

    @staticmethod
    def _bmm_flops(args: tuple[Any, ...]) -> tuple[int, int, int]:
        if (
            len(args) < 2
            or not isinstance(args[0], torch.Tensor)
            or not isinstance(args[1], torch.Tensor)
        ):
            return 0, 0, 0
        mat1, mat2 = args[0], args[1]
        if mat1.dim() != 3 or mat2.dim() != 3:
            return 0, 0, 0
        matrix_flops = 2 * mat1.shape[0] * mat1.shape[1] * mat1.shape[2] * mat2.shape[2]
        return matrix_flops, 0, matrix_flops

    def _matmul_flops(self, args: tuple[Any, ...], output: Any) -> tuple[int, int, int]:
        if (
            len(args) < 2
            or not isinstance(args[0], torch.Tensor)
            or not isinstance(args[1], torch.Tensor)
        ):
            return 0, 0, 0
        mat1, mat2 = args[0], args[1]
        if mat1.dim() == 0 or mat2.dim() == 0:
            return 0, 0, 0
        output_tensor = self._first_tensor(output)
        if output_tensor is None:
            return 0, 0, 0
        matrix_flops = 2 * output_tensor.numel() * mat1.shape[-1]
        return matrix_flops, 0, matrix_flops

    def _linear_flops(self, args: tuple[Any, ...], output: Any) -> tuple[int, int, int]:
        if (
            len(args) < 2
            or not isinstance(args[0], torch.Tensor)
            or not isinstance(args[1], torch.Tensor)
        ):
            return 0, 0, 0
        input_tensor, weight = args[0], args[1]
        if input_tensor.dim() == 0 or weight.dim() != 2:
            return 0, 0, 0
        output_elements = self._floating_output_elements(output)
        matrix_flops = 2 * output_elements * input_tensor.shape[-1]
        has_bias = len(args) > 2 and isinstance(args[2], torch.Tensor)
        exact_flops = matrix_flops + (output_elements if has_bias else 0)
        return exact_flops, 0, matrix_flops

    @staticmethod
    def _grouped_mm_flops(args: tuple[Any, ...]) -> tuple[int, int, int]:
        if (
            len(args) < 2
            or not isinstance(args[0], torch.Tensor)
            or not isinstance(args[1], torch.Tensor)
        ):
            return 0, 0, 0
        input_tensor, weight = args[0], args[1]
        if weight.dim() != 3 or input_tensor.dim() not in {2, 3}:
            return 0, 0, 0
        total_tokens = (
            input_tensor.shape[0] * input_tensor.shape[1]
            if input_tensor.dim() == 3
            else input_tensor.shape[0]
        )
        matrix_flops = 2 * total_tokens * weight.shape[1] * weight.shape[2]
        return matrix_flops, 0, matrix_flops

    def _scaled_mm_flops(
        self,
        args: tuple[Any, ...],
        output: Any,
    ) -> tuple[int, int, int]:
        if len(args) < 2 or not isinstance(args[0], torch.Tensor):
            return 0, 0, 0
        input_tensor = args[0]
        if input_tensor.dim() < 2:
            return 0, 0, 0
        output_elements = self._floating_output_elements(output)
        matrix_flops = 2 * output_elements * input_tensor.shape[-1]
        return matrix_flops, 0, matrix_flops

    def _attention_flops(
        self,
        op_name: str,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> tuple[int, int, int]:
        if len(args) < 3 or not all(isinstance(arg, torch.Tensor) for arg in args[:3]):
            return 0, 0, 0
        query, key, value = args[:3]
        if query.dim() < 2 or key.dim() < 2 or value.dim() < 2:
            return 0, 0, 0
        query_length, key_length = query.shape[-2], key.shape[-2]
        heads_per_batch = query.numel() // (query_length * query.shape[-1])
        scores_per_head = (
            self._causal_score_count(query_length, key_length)
            if self._is_causal_attention(op_name, args, kwargs)
            else query_length * key_length
        )
        score_count = heads_per_batch * scores_per_head
        matrix_flops = 2 * score_count * (query.shape[-1] + value.shape[-1])
        return matrix_flops, 5 * score_count, matrix_flops

    def _memory_bytes_for_op(  # noqa: PLR0911
        self,
        op_name: str,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
        output: Any,
    ) -> int:
        if self._matches_any(op_name, self._NO_DATA_MOVE_CREATION_OPS):
            return 0
        if self._matches_any(op_name, self._OUTPUT_WRITE_CREATION_OPS):
            return self._value_bytes(output)
        if self._matches_op(op_name, "aten::copy_"):
            if len(args) > 1:
                return self._value_bytes(args[1]) + self._value_bytes(output)
            return self._value_bytes(output)
        if self._matches_op(op_name, "aten::embedding"):
            return self._embedding_memory_bytes(args, output)
        if self._matches_op(op_name, "aten::index"):
            return self._indexed_read_memory_bytes(args, 1, output)
        if self._matches_any(op_name, {"aten::gather", "aten::index_select"}):
            return self._indexed_read_memory_bytes(args, 2, output)
        if self._matches_op(op_name, "aten::take"):
            return self._indexed_read_memory_bytes(args, 1, output)
        return (
            self._value_bytes(args)
            + self._value_bytes(kwargs)
            + self._value_bytes(output)
        )

    def _embedding_memory_bytes(self, args: tuple[Any, ...], output: Any) -> int:
        if len(args) < 2:
            return self._value_bytes(args) + self._value_bytes(output)
        output_bytes = self._value_bytes(output)
        return self._value_bytes(args[1]) + output_bytes + output_bytes

    def _indexed_read_memory_bytes(
        self,
        args: tuple[Any, ...],
        index_position: int,
        output: Any,
    ) -> int:
        if len(args) <= index_position:
            return self._value_bytes(args) + self._value_bytes(output)
        output_bytes = self._value_bytes(output)
        source_bytes = self._value_bytes(args[0])
        index_value = args[index_position]
        selected_source_bytes = (
            source_bytes if self._contains_bool_tensor(index_value) else output_bytes
        )
        return self._value_bytes(index_value) + selected_source_bytes + output_bytes

    def _is_alias_operation(
        self,
        op_name: str,
        args: tuple[Any, ...],
        output: Any,
    ) -> bool:
        if self._matches_any(op_name, self._ALIAS_OPS):
            return True
        if self._matches_any(op_name, {"aten::reshape", "aten::contiguous"}):
            input_tensor = self._first_tensor(args)
            output_tensor = self._first_tensor(output)
            return (
                input_tensor is not None
                and output_tensor is not None
                and self._shares_storage(input_tensor, output_tensor)
            )
        return False

    def _should_report_unaccounted_flops(
        self,
        op_name: str,
        output: Any,
        *,
        is_alias: bool,
        exact_flops: int,
        estimated_flops: int,
    ) -> bool:
        return (
            not is_alias
            and exact_flops == 0
            and estimated_flops == 0
            and self._contains_floating_tensor(output)
            and not self._matches_any(op_name, self._ZERO_FLOP_OPS)
            and not self._matches_any(op_name, self._NO_DATA_MOVE_CREATION_OPS)
            and not self._matches_any(op_name, self._OUTPUT_WRITE_CREATION_OPS)
        )

    @staticmethod
    def _matches_op(op_name: str, base_name: str) -> bool:
        return op_name == base_name or op_name.startswith(f"{base_name}.")

    @classmethod
    def _matches_any(cls, op_name: str, base_names: frozenset[str] | set[str]) -> bool:
        return any(cls._matches_op(op_name, base_name) for base_name in base_names)

    @staticmethod
    def _scalar_argument(
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
        position: int,
        name: str,
        default: float,
    ) -> float:
        value = kwargs.get(name, args[position] if len(args) > position else default)
        if isinstance(value, torch.Tensor):
            return float(value.item()) if value.numel() == 1 else default
        return float(value) if isinstance(value, (int, float)) else default

    @staticmethod
    def _causal_score_count(query_length: int, key_length: int) -> int:
        shared_length = min(query_length, key_length)
        return (
            shared_length * (shared_length + 1) // 2
            + max(query_length - key_length, 0) * key_length
        )

    @staticmethod
    def _is_causal_attention(
        op_name: str,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> bool:
        is_causal = kwargs.get("is_causal")
        if isinstance(is_causal, bool):
            return is_causal
        if "overrideable" in op_name and len(args) > 5 and isinstance(args[5], bool):
            return args[5]
        if "flash" in op_name and len(args) > 4 and isinstance(args[4], bool):
            return args[4]
        if len(args) > 5 and isinstance(args[5], bool):
            return args[5]
        return False

    @staticmethod
    def _iter_tensors(value: Any) -> Iterator[torch.Tensor]:
        if isinstance(value, torch.Tensor):
            yield value
        elif isinstance(value, Mapping):
            for item in value.values():
                yield from MetricCount._iter_tensors(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                yield from MetricCount._iter_tensors(item)

    @classmethod
    def _first_tensor(cls, value: Any) -> torch.Tensor | None:
        return next(cls._iter_tensors(value), None)

    @classmethod
    def _contains_floating_tensor(cls, value: Any) -> bool:
        return any(
            tensor.is_floating_point() or tensor.is_complex()
            for tensor in cls._iter_tensors(value)
        )

    @classmethod
    def _contains_bool_tensor(cls, value: Any) -> bool:
        return any(tensor.dtype is torch.bool for tensor in cls._iter_tensors(value))

    @classmethod
    def _floating_output_elements(cls, output: Any) -> int:
        return sum(
            tensor.numel()
            for tensor in cls._iter_tensors(output)
            if tensor.is_floating_point() or tensor.is_complex()
        )

    @classmethod
    def _value_bytes(cls, value: Any) -> int:
        return sum(cls._tensor_bytes(tensor) for tensor in cls._iter_tensors(value))

    @staticmethod
    def _tensor_bytes(tensor: torch.Tensor) -> int:
        element_count = tensor.numel()
        for size, stride in zip(tensor.shape, tensor.stride(), strict=True):
            if stride == 0 and size > 0:
                element_count //= size
        return element_count * tensor.element_size()

    @staticmethod
    def _shares_storage(first: torch.Tensor, second: torch.Tensor) -> bool:
        try:
            return (
                first.untyped_storage().data_ptr()
                == second.untyped_storage().data_ptr()
            )
        except RuntimeError:
            return False

    @staticmethod
    def _format_flops(value: int) -> str:
        """Format a FLOP count with an SI prefix and its exact value.

        Returns:
            Human-readable FLOPs followed by the exact count.
        """
        return (
            f"{humanize.metric(value, unit='FLOPs')} ({humanize.intcomma(value)} FLOPs)"
        )

    @staticmethod
    def _format_bytes(value: int) -> str:
        """Format a byte count with a binary prefix and its exact value.

        Returns:
            Human-readable bytes followed by the exact count.
        """
        return (
            f"{humanize.naturalsize(value, binary=True)} ({humanize.intcomma(value)} B)"
        )


def run_molmoact2_metric_count(
    device: str = DEFAULT_DEVICE,
    *,
    compile_model: bool = DEFAULT_COMPILE,
) -> tuple[torch.Tensor, MetricCount]:
    """Run one MolmoAct2 action prediction and return its metrics.

    Preprocessing remains outside the counter so the results describe model
    inference from backbone-ready tensors through action generation.

    Returns:
        The generated action chunk and metrics for its model inference.

    Raises:
        RuntimeError: If MolmoAct2 did not initialize its model components.
    """
    from physicalai.data import Feature, FeatureType, Observation  # noqa: PLC0415
    from physicalai.policies import MolmoAct2  # noqa: PLC0415

    torch.manual_seed(42)
    observation = Observation(
        images={
            "overview": torch.rand(1, 3, 256, 256),
            "wrist": torch.rand(1, 3, 256, 256),
        },
        state=torch.rand(1, 6),
        task=["example, input"],
    ).to(device)
    input_features = [
        Feature(name="overview", ftype=FeatureType.VISUAL, shape=(3, 256, 256)),
        Feature(name="state", ftype=FeatureType.STATE, shape=(6,)),
    ]
    output_features = [Feature(name="action", ftype=FeatureType.ACTION, shape=(6,))]
    policy = MolmoAct2(
        input_features=input_features,
        output_features=output_features,
        torch_compile=compile_model,
        load_weights=False,
    ).to(device=device, dtype=torch.bfloat16)
    policy.eval()

    with torch.no_grad(), MetricCount() as metrics:
        actions = policy.predict_action_chunk(observation)
    # from torch.utils.flop_counter import FlopCounterMode

    # with FlopCounterMode(depth=None) as flop_metrics:
    #     actions = model.predict_action_chunk(processed_batch)
    # print(f"FLOPs: {flop_metrics.get_total_flops()}")
    return actions, metrics


def main() -> None:
    """Run the MolmoAct2 metric counter from the command line."""
    actions, metrics = run_molmoact2_metric_count()
    print(f"Action shape: {tuple(actions.shape)}")  # noqa: T201
    print(metrics.format_report())  # noqa: T201


# class PrintingMode(TorchDispatchMode):
#     def __torch_dispatch__(self, func, types, args=(), kwargs=None):
#         print(func.name())
#         return func(*args, **kwargs)


if __name__ == "__main__":
    main()
    # with torch.device("xpu"), torch.no_grad():
    #     a = torch.rand(1, 3, 256, 256)
    #     b = torch.rand(1, 3, 256, 256)

    #     @torch.compile(mode="max-autotune")
    #     def f(x, y):
    #         return x + y * 2

    #     with PrintingMode():
    #         f(a, b)

    #     print("Compiled function:")
    #     with PrintingMode():
    #         f(a, b)
