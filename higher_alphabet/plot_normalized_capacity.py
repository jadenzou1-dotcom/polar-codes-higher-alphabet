# plot_normalized_capacity.py

import numpy as np
import matplotlib.pyplot as plt
import os

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "capacity_cache")
FORCE_RECOMPUTE = False

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


def estimate_normalized_capacities(N, p, q, trials):
    capacities = np.zeros(N, dtype=np.float64)

    for i in range(N):
        conditional_entropy_sum = 0.0

        for _ in range(trials):
            u = np.random.randint(0, q, size=N, dtype=np.int64)

            x = polar_transform(u, q)
            y = qary_channel(x, p, q)

            posterior = estimate_subchannel_prob_vector(
                y=y,
                true_u=u,
                symbol_index=i,
                p=p,
                q=q,
            )

            entropy = -np.sum(
                posterior * np.log2(np.maximum(posterior, 1e-300))
            )

            conditional_entropy_sum += entropy

        conditional_entropy = conditional_entropy_sum / trials

        # H(U_i) = log2(q), assuming uniform input
        mutual_information = np.log2(q) - conditional_entropy

        # Normalize to [0, 1]
        capacities[i] = mutual_information / np.log2(q)

        print(
            f"q={q}, subchannel={i:3d}, "
            f"normalized capacity={capacities[i]:.4f}"
        )

    return capacities

def theoretical_normalized_capacity(q, p):
    if p == 0:
        return 1.0

    H2 = -p * np.log2(p) - (1 - p) * np.log2(1 - p)

    C = np.log2(q) - H2 - p * np.log2(q - 1)

    return C / np.log2(q)

def theoretical_normalized_capacity(q, p):
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

    for q in Q_VALUES:
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
            label=f"q={q}",
        )

        # Optional: faint raw points
        plt.scatter(
            x_vals,
            y_vals,
            s=8,
            alpha=0.25,
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

        print(
            f"q={q}: C_norm={C_norm:.4f}, "
            f"boundary={boundary_k:.2f}, "
            f"area={total_area:.2f}, "
            f"area/N={normalized_area:.4f}"
        )
        # Since capacities are sorted low-to-high,
        # left side is frozen/bad and right side is usable/good.
        boundary_k = N * (1 - C_norm)

        plt.axvline(
            boundary_k,
            color=line.get_color(),
            linestyle="--",
            linewidth=1.5,
            alpha=0.5,
        )

        print(
            f"q={q}: theoretical normalized capacity={C_norm:.4f}, "
            f"boundary k={boundary_k:.1f}"
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
    plt.show()


if __name__ == "__main__":
    plot_capacity_trends()