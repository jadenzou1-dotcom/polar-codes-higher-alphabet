import numpy as np
import matplotlib.pyplot as plt

from polar_BSC import (
    polar_transform,
    bsc,
    bsc_subchannels_monte_carlo,
    freeze_bad_channels,
    bsc_obs_to_lr,
    sc_decode_lr,
)

N = 512
p = 0.2
TRIALS = 400
CONSTRUCTION_TRIALS = 800

def run_trial(N, p, K, frozen):
    u = np.zeros(N, dtype=np.int64)
    if K > 0:
        u[~frozen] = np.random.randint(0, 2, K)

    x = polar_transform(u)
    y = bsc(x, p)
    lr_vec = bsc_obs_to_lr(y, p)
    u_hat = sc_decode_lr(lr_vec, frozen)

    info_mask = ~frozen
    block_error = int(np.any(u_hat[info_mask] != u[info_mask]))
    return block_error

def run_experiment(N, p, trials, construction_trials):
    #Estimate bit-channel reliabilities once for this (N, p).
    metrics = bsc_subchannels_monte_carlo(
        N,
        p,
        trials=construction_trials,
    )

    K_values = np.arange(1, N + 1)
    error_probs = []

    for K in K_values:
        frozen = freeze_bad_channels(metrics, K)
        errors = 0

        for _ in range(trials):
            errors += run_trial(N, p, K, frozen)

        p_error = errors / trials
        error_probs.append(p_error)

        print(f"K={K:3d}, rate={K/N:.3f}, error={p_error:.6f}")

    return K_values, np.array(error_probs), metrics

if __name__ == "__main__":
    K_vals, err, metrics = run_experiment(
        N,
        p,
        TRIALS,
        CONSTRUCTION_TRIALS,
    )

    plt.figure()
    plt.plot(K_vals, err, marker="o")
    plt.xlabel("Number of information bits K")
    plt.ylabel("Block error probability")
    plt.title(
        f"Polar Code over BSC (N={N}, p={p}), {TRIALS} trials\n"
        f"Construction via Monte Carlo ({CONSTRUCTION_TRIALS} trials)"
    )
    plt.grid(True)
    plt.show()

#debug decoder for small block legnth to check if its implement ML
