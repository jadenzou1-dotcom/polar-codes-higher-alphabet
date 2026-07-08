import numpy as np
import matplotlib.pyplot as plt

# import everything from other file 
from polar_bec import (
    polar_transform,
    bec,
    bec_subchannels,
    freeze_bad_channels,
    bec_obs_to_lr,
    sc_decode_lr,
)

np.random.seed(0)

N = 128
e = 0.2
TRIALS = 200

#1 trial
def run_trial(N, e, K):
    Z = bec_subchannels(N, e)
    frozen = freeze_bad_channels(Z, K)

    u = np.zeros(N, dtype=np.int64)
    if K > 0:
        u[~frozen] = np.random.randint(0, 2, K)

    x = polar_transform(u)
    y = bec(x, e)
    lr_vec = bec_obs_to_lr(y)
    u_hat = sc_decode_lr(lr_vec, frozen)

    info_mask = ~frozen

    block_error = int(np.any(u_hat[info_mask] != u[info_mask]))

    return block_error

#monte carlo for all K values
def run_experiment(N, e, trials):
    K_values = np.arange(1, N + 1)
    error_probs = []

    for K in K_values:
        errors = 0

        for _ in range(trials):
            errors += run_trial(N, e, K)

        p_error = errors / trials
        error_probs.append(p_error)

        print(f"K={K:2d}, rate={K/N:.3f}, error={p_error:.6f}")

    return K_values, np.array(error_probs)


#run and plot
if __name__ == "__main__":
    K_vals, err = run_experiment(N, e, TRIALS)

    plt.figure()
    plt.plot(K_vals, err, marker="o")
    plt.xlabel("Number of information bits K")
    plt.ylabel("Block error probability")
    plt.title(f"Polar Code over BEC (N={N}, e={e}), {TRIALS} trials")
    plt.grid(True)
    plt.show()

    #error bar, x interval hover, local/global variable problem, data point size
    #qn: n=16, error for k=9 was lower than k=8, why