import time

import torch

from physicalai.data import Feature, FeatureType, Observation
from physicalai.policies import MolmoAct2


DEVICE = "cuda"
COMPILE = True
NUM_WARMUP = 5
NUM_ITERS = 50

torch.manual_seed(42)

# SO100 pretrained stats expect state-only inputs with 6 values.
batch = Observation(
    images={
        "overview": torch.rand(1, 3, 256, 256),
        "wrist": torch.rand(1, 3, 256, 256),
    },
    state=torch.rand(1, 6),
    task=["example, input"],
).to(DEVICE)

input_features = [
    Feature(name="overview", ftype=FeatureType.VISUAL, shape=(3, 256, 256)),
    Feature(name="state", ftype=FeatureType.STATE, shape=(6,)),
]
output_features = [
    Feature(name="action", ftype=FeatureType.ACTION, shape=(6,)),
]


def sync():
    if DEVICE == "xpu":
        torch.xpu.synchronize()
    elif DEVICE == "cuda":
        torch.cuda.synchronize()


def test_molmoact2(policy: MolmoAct2):
    print("running warmup...")
    with torch.no_grad():
        # Warmup (compilation, caching, lazy init, etc.)
        for _ in range(NUM_WARMUP):
            _ = policy.predict_action_chunk(batch)
        sync()
        print("finshed warmup...")

        print("running latency test...")
        # Timed runs
        latencies_ms = []
        for _ in range(NUM_ITERS):
            start = time.perf_counter()
            actions = policy.predict_action_chunk(batch)
            sync()
            end = time.perf_counter()
            latencies_ms.append((end - start) * 1000)

    latencies_ms.sort()
    n = len(latencies_ms)
    mean_ms = sum(latencies_ms) / n
    p50 = latencies_ms[n // 2]
    p90 = latencies_ms[int(n * 0.9)]
    p99 = latencies_ms[min(int(n * 0.99), n - 1)]

    print(f"Actions shape: {actions.shape}")
    print(f"\nLatency over {n} iterations (warmup={NUM_WARMUP}):")
    print(f"  mean: {mean_ms:.2f} ms")
    print(f"  p50:  {p50:.2f} ms")
    print(f"  p90:  {p90:.2f} ms")
    print(f"  p99:  {p99:.2f} ms")
    print(f"  min:  {latencies_ms[0]:.2f} ms")
    print(f"  max:  {latencies_ms[-1]:.2f} ms")


if __name__ == "__main__":

    policy = MolmoAct2(
        input_features=input_features,
        output_features=output_features,
        torch_compile=COMPILE,
        load_weights=False,
    ).to(device=DEVICE, dtype=torch.bfloat16)
    policy.eval()

    test_molmoact2(policy)
