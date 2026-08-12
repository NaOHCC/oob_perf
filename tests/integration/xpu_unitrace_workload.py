# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Collect a tiny XPU workload under an external unitrace wrapper."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from test.perf_analysis import CollectionConfig, HardwareSpec, collect_callable


def main() -> None:
    """Collect aligned profiler and unitrace artifacts for integration testing."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    left = torch.randn(64, 128, device="xpu", dtype=torch.float16)
    right = torch.randn(128, 32, device="xpu", dtype=torch.float16)
    collect_callable(
        lambda: torch.relu(left @ right),
        workload_name="xpu-unitrace-mm",
        hardware=HardwareSpec("XPU FP16 integration", 1.0, 1.0),
        config=CollectionConfig(
            device="xpu",
            output_dir=args.output_dir,
            warmup_iterations=1,
            measurement_iterations=2,
        ),
    )


if __name__ == "__main__":
    main()
