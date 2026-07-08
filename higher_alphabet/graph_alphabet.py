import numpy as np
import matplotlib.pyplot as plt

from alphabet_bsc_fast import (
    polar_transform,
    qary_channel,
    qary_sc_decode,
    qary_subchannels_monte_carlo,
    freeze_bad_channels,
)

N = 256
p = 0.1
q = 5

TRIALS = 200
CONSTRUCTION_TRIALS = 400


def run_trial(N, p, q, K, frozen, info_mask):
    u = np.zeros(N, dtype=np.int64)

    if K > 0:
        u[~frozen] = np.random.randint(0, q, K)

    x = polar_transform(u, q)
    y = qary_channel(x, p, q)

    u_hat = qary_sc_decode(y, frozen, N, p, q)
    block_error = int(np.any(u_hat[info_mask] != u[info_mask]))

    return block_error


def run_experiment(N, p, q, trials, construction_trials):
    metrics = qary_subchannels_monte_carlo(
        N,
        p,
        q,
        trials=construction_trials,
    )

    K_values = np.arange(1, N + 1)
    error_probs = []

    for K in K_values:
        frozen = freeze_bad_channels(metrics, K)
        info_mask = ~frozen

        errors = 0

        for _ in range(trials):
            errors += run_trial(N, p, q, K, frozen, info_mask)

        p_error = errors / trials
        error_probs.append(p_error)

        print(f"K={K:3d}, rate={K/N:.3f}, error={p_error:.6f}")

    return K_values, np.array(error_probs), metrics


if __name__ == "__main__":
    K_vals, err, metrics = run_experiment(
        N,
        p,
        q,
        TRIALS,
        CONSTRUCTION_TRIALS,
    )

    plt.figure()
    plt.plot(K_vals, err, marker="o")
    plt.xlabel("Number of information symbols K")
    plt.ylabel("Block error probability")
    plt.title(
        f"Polar Code over q-ary symmetric channel "
        f"(N={N}, q={q}, p={p}), {TRIALS} trials\n"
        f"Construction via Monte Carlo ({CONSTRUCTION_TRIALS} trials)"
    )
    plt.grid(True)
    plt.show()