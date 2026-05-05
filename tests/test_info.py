import numpy as np
import time
from scalib.modeling import RLDAClassifier
import os

# os.environ["RAYON_NUM_THREADS"] = "128"

NB = 32
NS = 3
P = 3
NV = 1
N_TRAIN = 10_000
N_EVAL = 10
SNR = 1

rng = np.random.default_rng(42)
signal_coefs = rng.normal(0, SNR, (NB, NS)).astype(np.float32)


def make_traces(labels, nb, ns, signal_coefs, noise_std=1):
    noise = rng.integers(-noise_std, noise_std, (len(labels), ns), dtype=np.int16)
    bits = ((labels[:, np.newaxis] >> np.arange(nb, dtype=np.uint64)) & 1).astype(
        np.float32
    )
    bits = bits * 2 - 1
    signal = (bits @ signal_coefs).astype(np.int16)
    return (noise + signal).astype(np.int16)


def test_get_info_correctness():
    print("\n--- correctness tests ---")
    rng_test = np.random.default_rng(123)

    for nb in [4, 8, 12]:
        ns = 3
        p = min(nb, 3)
        nc = 2**nb
        n_train = 5000
        n_test = 10
        noise = 500

        sc = rng_test.normal(0, 1.0, (nb, ns)).astype(np.float32)

        def make_traces_local(labels):
            noise_arr = rng_test.integers(
                -noise, noise, (len(labels), ns), dtype=np.int16
            )
            bits = (
                (labels[:, np.newaxis] >> np.arange(nb, dtype=np.uint64)) & 1
            ).astype(np.float32)
            bits = bits * 2 - 1
            signal = (bits @ sc).astype(np.int16)
            return (noise_arr + signal).astype(np.int16)

        # Train
        train_labels = rng_test.integers(0, nc, n_train, dtype=np.uint64)
        traces = make_traces_local(train_labels)
        rlda = RLDAClassifier(nb, p)
        rlda.fit_u(traces, train_labels[:, np.newaxis])
        rlda.solve()

        # Eval
        test_labels = rng_test.integers(0, nc, n_test, dtype=np.uint64)
        test_traces = make_traces_local(test_labels)

        # Reference from predict_proba
        prs = rlda.predict_proba(test_traces, 0)
        ref = np.log2(prs[np.arange(n_test), test_labels])

        # get_info
        info = rlda.get_info(test_traces, test_labels, 0)

        max_err = np.abs(info - ref).max()
        mean_err = np.abs(info - ref).mean()
        # get_info
        info = rlda.get_info(test_traces, test_labels, 0)
        print(f"Delta for {nb} bits : {ref-info}")


def test_get_info_benchmark():
    print(f"\n--- benchmark nb=32, {N_EVAL} traces ---")

    # Setup
    train_labels = rng.integers(0, 2**NB, N_TRAIN, dtype=np.uint64)
    traces = make_traces(train_labels, NB, NS, signal_coefs)
    rlda = RLDAClassifier(NB, P)
    rlda.fit_u(traces, train_labels[:, np.newaxis])
    rlda.solve()

    eval_labels = rng.integers(0, 2**NB, N_EVAL, dtype=np.uint64)
    eval_traces = make_traces(eval_labels, NB, NS, signal_coefs)

    # Warmup
    tmp_res = rlda.get_info(eval_traces, eval_labels, 0)

    # Benchmark
    N_REPEATS = 1
    times = []
    for _ in range(N_REPEATS):
        t0 = time.perf_counter()
        result = rlda.get_info(eval_traces, eval_labels, 0)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    print(f"mean: {np.mean(times):.3f}s  " f"min: {np.min(times):.3f}s")
    print(f"PI estimate: {NB + np.mean(result):.4f} bits")


if __name__ == "__main__":
    test_get_info_correctness()
    test_get_info_benchmark()
