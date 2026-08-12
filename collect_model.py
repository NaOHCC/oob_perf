# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Collect MolmoAct2 performance on a configured XPU or CUDA device."""

import argparse
from pathlib import Path

import torch
from perf_analysis import collect_callable, CollectionConfig, HardwareSpec
from physicalai.data import Feature, FeatureType, Observation
from physicalai.policies import MolmoAct2

DEFAULT_COMPILE = False


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect MolmoAct2 performance")
    parser.add_argument("--device", choices=("cuda", "xpu"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hardware-label", required=True)
    parser.add_argument("--peak-tflops", type=float, required=True)
    parser.add_argument("--memory-bandwidth-gbs", type=float, required=True)
    parser.add_argument(
        "--unitrace",
        action="store_true",
        help="Require an external unitrace wrapper (XPU only)",
    )
    return parser


def main() -> None:
    """Build MolmoAct2 and collect one configured performance run."""
    args = _parser().parse_args()
    torch.manual_seed(42)
    observation = Observation(
        images={
            "overview": torch.rand(1, 3, 256, 256),
            "wrist": torch.rand(1, 3, 256, 256),
        },
        state=torch.rand(1, 6),
        task=["example, input"],  # type: ignore[arg-type]
    ).to(args.device)
    input_features = [
        Feature(name="overview", ftype=FeatureType.VISUAL, shape=(3, 256, 256)),
        Feature(name="state", ftype=FeatureType.STATE, shape=(6,)),
    ]
    output_features = [
        Feature(name="action", ftype=FeatureType.ACTION, shape=(6,)),
    ]
    policy = MolmoAct2(
        input_features=input_features,
        output_features=output_features,
        torch_compile=DEFAULT_COMPILE,
        load_weights=False,
    ).to(device=args.device, dtype=torch.bfloat16)
    policy.eval()

    with torch.no_grad():
        collect_callable(
            lambda: policy.predict_action_chunk(observation),
            workload_name="molmoact2",
            hardware=HardwareSpec(
                label=args.hardware_label,
                peak_tflops=args.peak_tflops,
                memory_bandwidth_gbs=args.memory_bandwidth_gbs,
            ),
            config=CollectionConfig(
                device=args.device,
                output_dir=args.output_dir,
                require_unitrace=args.unitrace,
            ),
        )


if __name__ == "__main__":
    main()
