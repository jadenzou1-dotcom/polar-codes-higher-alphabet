# plot_normalized_capacity.py

import numpy as np
import matplotlib.pyplot as plt
import os
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "capacity_cache")
RESULTS_DIR = os.path.join(HERE, "results")
FORCE_RECOMPUTE = False
SEED = 0               # Base seed so reruns reproduce the cached results

from alphabet_bsc_fast import (
    polar_transform,
    qary_channel,
    qary_obs_to_probs,
    qary_f,
    qary_g,
)

N = 512
p = 0.1
Q_VALUES = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 50] 
#Q_VALUES = list(range(2, 26))
TRIALS = 500           # Increase for smoother curves
SORT_CAPACITIES = True # True = clean trend comparison, False = original subchannel order
SMOOTH_WINDOW = 7      # Increase for smoother trend line, use 1 for no smoothing


def moving_average(x, window):
    if window <= 1:
        return x

    out = np.zeros_like(x)

    half = window // 2

    for i in range(len(x)):
        start = max(0, i - half)
        end = min(len(x), i + half + 1)

        out[i] = np.mean(x[start:end])

    return out


def estimate_subchannel_prob_vector(y, true_u, symbol_index, p, q):
    """
    Estimate P(U_i = a | Y^N, U_1^{i-1}) for one received vector y.

    Uses genie-given previous symbols correctly.
    """

    obs_probs = qary_obs_to_probs(y, p, q)

    def recurse(obs_node, u_node, local_index):
        n = len(obs_node)

        if n == 1:
            probs = obs_node[0]
            return probs / np.sum(probs)

        half = n // 2

        if local_index < half:
            # Target is in the upper branch.
            # Lower branch is unknown, so f marginalizes it out.
            upper_obs = np.zeros((half, q), dtype=np.float64)

            for j in range(half):
                upper_obs[j] = qary_f(
                    obs_node[2 * j],
                    obs_node[2 * j + 1],
                    q,
                )

            return recurse(
                upper_obs,
                u_node[:half],
                local_index,
            )

        else:
            # Target is in the lower branch.
            # The entire upper branch is previous information,
            # so it is genie-known.
            true_upper_u = u_node[:half]

            # IMPORTANT:
            # g needs the encoded upper partial sums, not raw u values.
            true_upper_x = polar_transform(true_upper_u, q)

            lower_obs = np.zeros((half, q), dtype=np.float64)

            for j in range(half):
                lower_obs[j] = qary_g(
                    obs_node[2 * j],
                    obs_node[2 * j + 1],
                    true_upper_x[j],
                    q,
                )

            return recurse(
                lower_obs,
                u_node[half:],
                local_index - half,
            )

    return recurse(obs_probs, true_u, symbol_index)


# ---------------------------------------------------------------------------
# Batched versions of the above. Same math as estimate_subchannel_prob_vector,
# qary_f, qary_g and polar_transform, but run on all trials at once
# (arrays have a leading trial axis) so numpy does the loops instead of Python.
# ---------------------------------------------------------------------------

def normalize_probs_batch(v):
    s = np.sum(v, axis=-1, keepdims=True)
    q = v.shape[-1]
    return np.where(s == 0, 1.0 / q, v / np.where(s == 0, 1.0, s))


def polar_transform_batch(u, q):
    n = u.shape[-1]
    if n == 1:
        return u

    x = np.zeros_like(u)
    x[..., :n // 2] = polar_transform_batch((u[..., ::2] + u[..., 1::2]) % q, q)
    x[..., n // 2:] = polar_transform_batch(u[..., 1::2], q)
    return x


def qary_f_batch(left, right, q):
    # out[a] = sum_b left[(a+b) % q] * right[b]
    out = np.zeros_like(left)
    for b in range(q):
        out += np.roll(left, -b, axis=-1) * right[..., b:b + 1]
    return normalize_probs_batch(out / q)


def qary_g_batch(left, right, upper_symbol, q):
    # out[b] = left[(upper_symbol + b) % q] * right[b]
    idx = (upper_symbol[..., None] + np.arange(q)) % q
    return normalize_probs_batch(np.take_along_axis(left, idx, axis=-1) * right)


def estimate_subchannel_prob_vectors_batch(y, true_u, symbol_index, p, q):
    trials, n = y.shape
    obs = np.full((trials, n, q), p / (q - 1), dtype=np.float64)
    np.put_along_axis(obs, y[..., None], 1 - p, axis=-1)
    obs = normalize_probs_batch(obs)

    u_node = true_u
    local_index = symbol_index

    while obs.shape[1] > 1:
        half = obs.shape[1] // 2
        left, right = obs[:, 0::2], obs[:, 1::2]

        if local_index < half:
            obs = qary_f_batch(left, right, q)
            u_node = u_node[:, :half]
        else:
            true_upper_x = polar_transform_batch(u_node[:, :half], q)
            obs = qary_g_batch(left, right, true_upper_x, q)
            u_node = u_node[:, half:]
            local_index -= half

    return obs[:, 0]


def subchannel_capacity(args):
    # One subchannel's normalized capacity; run in a worker process.
    N, p, q, trials, i, seed = args
    rng = np.random.default_rng(seed)

    u = rng.integers(0, q, size=(trials, N), dtype=np.int64)
    x = polar_transform_batch(u, q)

    flip = rng.random((trials, N)) < p
    e = rng.integers(1, q, size=(trials, N))
    y = np.where(flip, (x + e) % q, x)

    posterior = estimate_subchannel_prob_vectors_batch(y, u, i, p, q)
    entropy = -np.sum(posterior * np.log2(np.maximum(posterior, 1e-300)), axis=-1)

    # H(U_i) = log2(q), assuming uniform input
    mutual_information = np.log2(q) - np.mean(entropy)

    # Normalize to [0, 1]
    return mutual_information / np.log2(q)


def estimate_normalized_capacities(N, p, q, trials):
    seeds = np.random.SeedSequence([SEED, q]).spawn(N)
    jobs = [(N, p, q, trials, i, seeds[i]) for i in range(N)]

    with ProcessPoolExecutor() as pool:
        capacities = np.array(list(pool.map(subchannel_capacity, jobs, chunksize=4)))

    print(f"q={q}: computed {N} subchannel capacities")
    return capacities


def theoretical_normalized_capacity(q, p):
    if p == 0:
        return 1.0

    H2 = -p * np.log2(p) - (1 - p) * np.log2(1 - p)

    C = np.log2(q) - H2 - p * np.log2(q - 1)

    return C / np.log2(q)


def area_from_theoretical_step(x_vals, y_vals, boundary_k, N):
    # Add k=0 point so the area starts at 0
    x = np.concatenate(([0], x_vals))
    y = np.concatenate(([0], y_vals))

    # Add exact boundary point by interpolation
    y_boundary = np.interp(boundary_k, x, y)

    x_left = np.concatenate((x[x < boundary_k], [boundary_k]))
    y_left = np.concatenate((y[x < boundary_k], [y_boundary]))

    x_right = np.concatenate(([boundary_k], x[x > boundary_k]))
    y_right = np.concatenate(([y_boundary], y[x > boundary_k]))

    area_left = np.trapezoid(y_left, x_left)
    area_right = np.trapezoid(1 - y_right, x_right)

    total_area = area_left + area_right
    normalized_area = total_area / N

    return total_area, normalized_area

def plot_capacity_trends():
    plt.figure(figsize=(9, 6))
    area_results = []

    colors = plt.cm.viridis(np.linspace(0, 0.92, len(Q_VALUES)))

    for q, color in zip(Q_VALUES, colors):
        os.makedirs(CACHE_DIR, exist_ok=True)

        cache_file = os.path.join(
            CACHE_DIR,
            f"capacity_N{N}_p{p}_q{q}_trials{TRIALS}.npy"
        )

        if os.path.exists(cache_file) and not FORCE_RECOMPUTE:
            print(f"Loading saved capacities for q={q}")
            caps = np.load(cache_file)
        else:
            print(f"Computing capacities for q={q}")
            caps = estimate_normalized_capacities(N, p, q, TRIALS)
            np.save(cache_file, caps)
            print(f"Saved capacities for q={q}")
        print(f"\nq={q}")
        print(f"Average normalized capacity = {np.mean(caps):.4f}")
        print(f"Sum of normalized capacities = {np.sum(caps):.4f}")

        if SORT_CAPACITIES:
            y_vals = np.sort(caps)
            x_vals = np.arange(1, N + 1)
            x_label = "Sorted subchannel index k"
        else:
            y_vals = caps
            x_vals = np.arange(N)
            x_label = "Subchannel index k"

        trend = moving_average(y_vals, SMOOTH_WINDOW)

        line, = plt.plot(            
            x_vals,
            trend,
            linewidth=2,
            color=color,
            label=f"q={q}",
        )

        # Optional: faint raw points
        plt.scatter(
            x_vals,
            y_vals,
            s=8,
            alpha=0.25,
            color=color,
        )
        C_norm = theoretical_normalized_capacity(q, p)
        boundary_k = N * (1 - C_norm)

        plt.axvline(
            boundary_k,
            color=line.get_color(),
            linestyle="--",
            linewidth=1.5,
            alpha=0.6,
        )

        total_area, normalized_area = area_from_theoretical_step(
            x_vals,
            trend,
            boundary_k,
            N
        )

        area_results.append([
            q,
            C_norm,
            boundary_k,
            total_area,
            normalized_area
        ])

        # Since capacities are sorted low-to-high,
        # left side is frozen/bad and right side is usable/good.
        print(
            f"q={q}: C_norm={C_norm:.4f}, "
            f"boundary={boundary_k:.2f}, "
            f"area={total_area:.2f}, "
            f"area/N={normalized_area:.4f}"
        )

    plt.xlabel(x_label)
    plt.ylabel("Normalized subchannel capacity")
    plt.ylim(-0.05, 1.05)
    plt.title(
        f"Normalized Polarized Subchannel Capacity\n"
        f"N={N}, p={p}, trials={TRIALS}"
    )
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    print("\n" + "=" * 70)
    print("AREA SUMMARY")
    print("=" * 70)
    print(f"{'q':>4} {'C_norm':>10} {'boundary_k':>12} {'area':>12} {'area/N':>12}")
    print("-" * 70)

    for row in area_results:
        q, C_norm, boundary_k, total_area, normalized_area = row
        print(
            f"{q:>4} "
            f"{C_norm:>10.4f} "
            f"{boundary_k:>12.2f} "
            f"{total_area:>12.2f} "
            f"{normalized_area:>12.4f}"
        )

    os.makedirs(RESULTS_DIR, exist_ok=True)
    tag = f"N{N}_p{p}_trials{TRIALS}"
    plt.savefig(os.path.join(RESULTS_DIR, f"normalized_capacity_{tag}.png"), dpi=150)
    save_area_results(area_results, tag)
    plot_area_vs_q(area_results, tag)
    plt.show()


def save_area_results(area_results, tag):
    header = ["q", "C_norm", "boundary_k", "area", "area_over_N"]
    np.savetxt(
        os.path.join(RESULTS_DIR, f"area_summary_{tag}.csv"),
        np.array(area_results),
        delimiter=",",
        header=",".join(header),
        comments="",
        fmt=["%d", "%.4f", "%.2f", "%.2f", "%.4f"],
    )

    with open(os.path.join(RESULTS_DIR, f"area_summary_{tag}.md"), "w") as fh:
        fh.write(f"# Polarization speed (area to ideal step), N={N}, p={p}, trials={TRIALS}\n\n")
        fh.write("| q | normalized capacity | ideal step at k | area | area / N |\n")
        fh.write("|---:|---:|---:|---:|---:|\n")
        for q, C_norm, boundary_k, total_area, normalized_area in area_results:
            fh.write(
                f"| {q:d} | {C_norm:.4f} | {boundary_k:.2f} | "
                f"{total_area:.2f} | {normalized_area:.4f} |\n"
            )


def plot_area_vs_q(area_results, tag):
    rows = np.array(area_results)
    fig = plt.figure(figsize=(7, 4.5))
    plt.plot(rows[:, 0], rows[:, 4], marker="o")
    plt.xscale("log")
    plt.xticks(rows[:, 0], [f"{int(q)}" for q in rows[:, 0]])
    plt.minorticks_off()
    plt.xlabel("Alphabet size q (log scale)")
    plt.ylabel("Area to ideal step / N  (lower = faster polarization)")
    plt.title(f"Speed of polarization vs alphabet size\nN={N}, p={p}, trials={TRIALS}")
    plt.grid(True)
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, f"area_vs_q_{tag}.png"), dpi=150)


if __name__ == "__main__":
    plot_capacity_trends()