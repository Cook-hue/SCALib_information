import numpy as np
import time
from scalib.modeling import RLDAClassifier

NB = 20
NS = 3
P = 3
NV = 1
N_TRAIN = 10_000
N_EVAL = 100
SNR = 1.0  # signal amplitude, tune to get realistic SNR

# Build a simple linear signal model: each bit contributes to each sample
# coefs shape: (NB, NS) — one coefficient per bit per sample
rng = np.random.default_rng(42)
signal_coefs = rng.normal(0, SNR, (NB, NS)).astype(np.float32)


def make_traces(labels, n, noise_std=1000):
    """labels shape: (n,), returns traces shape: (n, NS)"""
    noise = rng.integers(-noise_std, noise_std, (n, NS), dtype=np.int16)
    # Extract bits: shape (n, NB)
    bits = ((labels[:, np.newaxis] >> np.arange(NB, dtype=np.uint64)) & 1).astype(
        np.float32
    )
    # Map 0→-1, 1→+1
    bits = bits * 2 - 1
    # Signal: (n, NB) @ (NB, NS) → (n, NS)
    signal = (bits @ signal_coefs).astype(np.int16)
    return (noise + signal).astype(np.int16)


# Setup — not timed
rlda = RLDAClassifier(NB, P)
train_labels = rng.integers(0, 2**NB, N_TRAIN, dtype=np.uint64)
traces = make_traces(train_labels, N_TRAIN)
rlda.fit_u(traces, train_labels[:, np.newaxis], 1)
rlda.solve()

eval_labels = rng.integers(0, 2**NB, N_EVAL, dtype=np.uint64)
eval_traces = make_traces(eval_labels, N_EVAL)


# Benchmark — only time this
N_REPEATS = 2
times = []
for _ in range(N_REPEATS):
    t0 = time.perf_counter()
    # your get_info call goes here
    rlda.get_info(eval_traces, eval_labels, 0)
    t1 = time.perf_counter()
    times.append(t1 - t0)

print(
    f"mean: {np.mean(times):.3f}s  std: {np.std(times):.3f}s  min: {np.min(times):.3f}s"
)
