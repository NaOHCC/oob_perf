from __future__ import annotations

import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, ClassVar

import humanize
import torch
import triton
from torch.profiler import itt
from torch.utils._python_dispatch import TorchDispatchMode  # noqa: PLC2701

DEFAULT_DEVICE = "xpu"
DEFAULT_COMPILE = False


@contextmanager
def ut_range():
    os.environ["PTI_ENABLE_COLLECTION"] = "1"
    try:
        yield
    finally:
        os.environ["PTI_ENABLE_COLLECTION"] = "0"


def run_molmoact2_metric_count(
    device: str = DEFAULT_DEVICE,
    *,
    compile_model: bool = DEFAULT_COMPILE,
):
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

    t = triton.testing.do_bench(lambda: policy.predict_action_chunk(observation))

    print(f"MolmoAct2 metric count on {device} (compile={compile_model}): {t} ms")

    with ut_range():
        import time

        t = time.time()
        policy.predict_action_chunk(observation)
        t = (time.time() - t) * 1000
        print(
            f"MolmoAct2 metric count on {device} (compile={compile_model}) [UT range]: {t} ms"
        )


def main() -> None:
    """Run the MolmoAct2 metric counter from the command line."""
    run_molmoact2_metric_count()


if __name__ == "__main__":
    main()
