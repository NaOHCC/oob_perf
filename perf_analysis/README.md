# Callable Performance Analysis

Analyze one repeatable PyTorch workload without the benchmark repository. The
tool combines logical ATen FLOPs and data movement, synchronized wall latency,
PyTorch profiler attribution, and XPU unitrace kernel timing.

## Collection

Create a script at the repository root that owns the model, inputs, and lambda:

```python
from pathlib import Path

import torch

from perf_analysis import (
    CollectionConfig,
    HardwareSpec,
    collect_callable,
)

model = torch.nn.Linear(4096, 4096, device="xpu", dtype=torch.float16).eval()
inputs = torch.randn(32, 4096, device="xpu", dtype=torch.float16)

collect_callable(
    lambda: model(inputs),
    workload_name="linear-fp16",
    hardware=HardwareSpec(
        label="Target GPU FP16",
        peak_tflops=100.0,
        memory_bandwidth_gbs=500.0,
    ),
    config=CollectionConfig(
        device="xpu",
        output_dir=Path("artifacts/linear-fp16"),
    ),
)
```

Run the script through unitrace. The output directory passed to unitrace must
match `CollectionConfig.output_dir`:

```bash
uv run --project library /path/to/unitrace \
  --chrome-kernel-logging \
  --start-paused \
  --output-dir-path artifacts/linear-fp16 \
  python collect_model.py
```

The callable runs for warmup, latency measurement, metric counting, and one
aligned profiler/unitrace trace. It must be repeatable. A stateful model, such
as a decoder with a KV cache, must restore its state inside the callable.

The tool does not change `eval` mode, gradient mode, random seeds, autocast, or
model state. The caller owns those choices.

## Reporting

After the unitrace-wrapped process exits, generate structured and human-readable
reports:

```bash
PYTHONPATH=test uv run --project library python -m perf_analysis \
  artifacts/linear-fp16/collection.json \
  --output-dir artifacts/linear-fp16
```

This writes `analysis.json` and `analysis.md` and prints a terminal summary.
Python callers can use `analyze_collection(...)`, `render_text(...)`, and
`render_markdown(...)` directly.

Strict unitrace mapping is the default. To inspect profiler-only results when
unitrace is unavailable or does not match, opt in explicitly:

```bash
PYTHONPATH=test uv run --project library python -m perf_analysis \
  artifacts/linear-fp16/collection.json \
  --allow-profiler-fallback
```

## Interpretation

For every dispatched operation, the projection computes:

```text
T1_op = max(FLOPs / peak FLOPs per second, logical bytes / DRAM bytes per second)
T1 = sum(T1_op)
R = T1 / median synchronized wall latency
```

`T2 device sum` is the sum of mapped kernel durations and is diagnostic. The
primary T2 and R use synchronized wall latency.

`MetricCount` reports logical tensor reads and writes. This is not a hardware
DRAM transaction count and does not model cache residency, fusion, or allocator
traffic. The supplied hardware peak must match the workload precision.

## XPU and A100 Comparison

Collect the same workload independently on each machine, then compare the two
`analysis.json` files offline. Keep model weights, inputs, batch size, dtype,
evaluation mode, and compile settings identical. The commands below run from
the repository root.

The MolmoAct2 example accepts the device and hardware roofline explicitly. XPU
can use either PyTorch profiler or unitrace. This profiler-only command does not
need a unitrace wrapper:

```bash
PYTHONPATH=test uv run --project library python test/collect_model.py \
  --device xpu \
  --output-dir test/artifacts/molmoact2/xpu \
  --hardware-label "PTL BF16" \
  --peak-tflops 58 \
  --memory-bandwidth-gbs 110
```

To use unitrace instead, add `--unitrace` and wrap the same command:

```bash
PYTHONPATH=test uv run --project library /path/to/unitrace \
  --chrome-kernel-logging \
  --start-paused \
  --output-dir-path test/artifacts/molmoact2/xpu \
  python test/collect_model.py \
  --device xpu \
  --output-dir test/artifacts/molmoact2/xpu \
  --hardware-label "PTL BF16" \
  --peak-tflops 58 \
  --memory-bandwidth-gbs 110 \
  --unitrace
```

On the A100 machine, provide the dense BF16 peak throughput and DRAM bandwidth
for the exact A100 SKU. Do not use sparse throughput. PCIe/SXM and 40/80 GB
models have different specifications.

```bash
export A100_PEAK_TFLOPS=<dense-bf16-tflops>
export A100_BANDWIDTH_GBS=<dram-bandwidth-gbs>

PYTHONPATH=test uv run --project library python test/collect_model.py \
  --device cuda \
  --output-dir test/artifacts/molmoact2/a100 \
  --hardware-label "A100 BF16" \
  --peak-tflops "$A100_PEAK_TFLOPS" \
  --memory-bandwidth-gbs "$A100_BANDWIDTH_GBS"
```

Generate each single-device report on the machine that owns its trace. XPU
profiler-only and CUDA are native profiler sources and do not need
`--allow-profiler-fallback`:

```bash
PYTHONPATH=test uv run --project library python -m perf_analysis \
  test/artifacts/molmoact2/xpu/collection.json \
  --output-dir test/artifacts/molmoact2/xpu

PYTHONPATH=test uv run --project library python -m perf_analysis \
  test/artifacts/molmoact2/a100/collection.json \
  --output-dir test/artifacts/molmoact2/a100
```

To use XPU unitrace data directly during comparison, keep the XPU
`collection.json`, profiler trace, and unitrace trace together. Copy the A100
`analysis.json` to that machine, then pass the XPU collection as the reference:

```bash
PYTHONPATH=test uv run --project library python -m perf_analysis.compare \
  test/artifacts/molmoact2/xpu/collection.json \
  test/artifacts/molmoact2/a100/analysis.json \
  --reference-unitrace test/artifacts/molmoact2/xpu/python.<pid>.json \
  --reference-name XPU \
  --target-name A100 \
  --output-dir test/artifacts/molmoact2/comparison
```

Omit `--reference-unitrace` to discover the trace through the collection
manifest's glob. Exactly one matching unitrace JSON must exist. Unitrace
mapping is strict by default; use `--allow-reference-profiler-fallback` only
when an explicitly reported XPU profiler fallback is acceptable. An XPU
collection captured without `--unitrace` cannot be retrofitted with a unitrace
trace because the collection interval was not aligned under the unitrace
contract.

The helper script exposes the same workflow:

```bash
cd test

./run.sh collect xpu ptl "PTL BF16" 58 110 --unitrace
./run.sh collect cuda a100 "A100 BF16" "$A100_PEAK_TFLOPS" "$A100_BANDWIDTH_GBS"
./run.sh compare ptl a100 artifacts/molmoact2/ptl/python.<pid>.json
```

Set `UNITRACE_BIN` when unitrace is not available at the script's default
workspace location. The compare command can omit its final path to use manifest
discovery.

The primary result is target wall speedup:

```text
A100 speedup over XPU = XPU wall latency / A100 wall latency
```

A value greater than one means A100 is faster. Each device's `R = T1 / T2`
uses that device's own peak throughput and bandwidth, so R describes proximity
to its own roofline rather than absolute cross-device speed. Per-operator
timings may come from different trace sources and are diagnostic; synchronized
wall latency is the end-to-end comparison.