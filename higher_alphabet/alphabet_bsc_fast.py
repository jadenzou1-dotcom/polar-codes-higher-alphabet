import numpy as np

N = 32
p = 0.05
K = 4
q = 3


# q-ary polar transform
# Same structure as the binary polar transform, but uses mod q instead of mod 2.
def polar_transform(u, q):
    if len(u) == 1:
        return u

    else:
        x = np.zeros(len(u), dtype=np.int64)

        x[:len(u)//2] = polar_transform((u[::2] + u[1::2]) % q, q)
        x[len(u)//2:] = polar_transform(u[1::2], q)

        return x


# q-ary symmetric channel
# With probability 1-p, receive the same symbol.
# With probability p, flip to one of the other q-1 symbols uniformly.
def qary_channel(x, p, q):
    x = np.asarray(x, dtype=np.int64)

    r = np.random.random_sample(len(x))
    e = np.random.randint(1, q, size=len(x))

    y = np.array(x, copy=True, dtype=np.int64)

    flip = r < p
    y[flip] = (y[flip] + e[flip]) % q

    return y


# Channel transition probability W(y|x).
# This does not simulate the channel. It tells the decoder how likely y is
# if x was transmitted.
def channel_prob(y, x, p, q):
    if y == x:
        return 1 - p
    else:
        return p / (q - 1)


# Keeps probability vectors numerically stable.
# This does not change the argmax, it just rescales the vector.
def normalize_probs(v):
    s = np.sum(v)

    if s == 0:
        return np.ones(len(v), dtype=np.float64) / len(v)

    return v / s


# q-ary version of the binary f update.
# Used for decoding the upper branch.
# It sums over the unknown lower symbol.
def qary_f(left_probs, right_probs, q):
    out = np.zeros(q, dtype=np.float64)

    for a in range(q):
        total = 0.0

        for b in range(q):
            total += left_probs[(a + b) % q] * right_probs[b]

        out[a] = total / q

    return normalize_probs(out)


# q-ary version of the binary g update.
# Used for decoding the lower branch.
# The upper symbol has already been decoded.
def qary_g(left_probs, right_probs, upper_symbol, q):
    out = np.zeros(q, dtype=np.float64)

    for b in range(q):
        out[b] = left_probs[(upper_symbol + b) % q] * right_probs[b]

    return normalize_probs(out)


# Convert received q-ary symbols into probability vectors.
# For each received symbol y[i], this creates:
# [P(y[i]|x=0), P(y[i]|x=1), ..., P(y[i]|x=q-1)]
def qary_obs_to_probs(y, p, q):
    y = np.asarray(y, dtype=np.int64)

    probs = np.zeros((len(y), q), dtype=np.float64)

    for i in range(len(y)):
        for a in range(q):
            probs[i, a] = channel_prob(y[i], a, p, q)

        probs[i] = normalize_probs(probs[i])

    return probs


# Fast q-ary successive cancellation decoder.
# This replaces the slow sc_symbol_probs approach.
# Instead of recomputing recursive likelihoods from scratch for each symbol,
# this passes q-length probability vectors through the polar tree.
def qary_sc_decode(y, frozen, N, p, q, frozen_values=None):
    obs_probs = qary_obs_to_probs(y, p, q)
    frozen = np.asarray(frozen, dtype=bool)

    if frozen_values is None:
        frozen_values = np.zeros(N, dtype=np.int64)
    else:
        frozen_values = np.asarray(frozen_values, dtype=np.int64)

    def _decode_node(obs_node, frozen_node, frozen_values_node):
        n = len(obs_node)

        if n == 1:
            if frozen_node[0]:
                symbol = frozen_values_node[0]
            else:
                symbol = np.argmax(obs_node[0])

            u_hat_node = np.array([symbol], dtype=np.int64)
            x_hat_node = np.array([symbol], dtype=np.int64)

            return u_hat_node, x_hat_node

        half = n // 2

        # Upper branch: q-ary f update
        upper_obs = np.zeros((half, q), dtype=np.float64)

        for i in range(half):
            upper_obs[i] = qary_f(
                obs_node[2 * i],
                obs_node[2 * i + 1],
                q,
            )

        u_hat_upper, x_hat_upper = _decode_node(
            upper_obs,
            frozen_node[:half],
            frozen_values_node[:half],
        )

        # Lower branch: q-ary g update
        lower_obs = np.zeros((half, q), dtype=np.float64)

        for i in range(half):
            lower_obs[i] = qary_g(
                obs_node[2 * i],
                obs_node[2 * i + 1],
                x_hat_upper[i],
                q,
            )

        u_hat_lower, x_hat_lower = _decode_node(
            lower_obs,
            frozen_node[half:],
            frozen_values_node[half:],
        )

        u_hat_node = np.zeros(n, dtype=np.int64)
        u_hat_node[:half] = u_hat_upper
        u_hat_node[half:] = u_hat_lower

        x_hat_node = np.zeros(n, dtype=np.int64)

        for i in range(half):
            x_hat_node[2 * i] = (x_hat_upper[i] + x_hat_lower[i]) % q
            x_hat_node[2 * i + 1] = x_hat_lower[i]

        return u_hat_node, x_hat_node

    u_hat, _ = _decode_node(obs_probs, frozen, frozen_values)

    return u_hat


# Monte Carlo construction for q-ary polar code.
# For each subchannel, only that symbol is treated as information.
# Previous symbols are genie-given by freezing them to their true values.
def qary_subchannels_monte_carlo(N, p, q, trials):
    error_probs = np.zeros(N, dtype=np.float64)

    for symbol_index in range(N):
        errors = 0

        frozen = np.ones(N, dtype=bool)
        frozen[symbol_index] = False

        for _ in range(trials):
            u = np.random.randint(0, q, size=N, dtype=np.int64)

            x = polar_transform(u, q)
            y = qary_channel(x, p, q)

            frozen_values = np.zeros(N, dtype=np.int64)
            frozen_values[:symbol_index] = u[:symbol_index]

            u_hat = qary_sc_decode(
                y,
                frozen,
                N,
                p,
                q,
                frozen_values=frozen_values,
            )

            if u_hat[symbol_index] != u[symbol_index]:
                errors += 1

        error_probs[symbol_index] = errors / trials

    return error_probs


def freeze_bad_channels(metric, K):
    n = len(metric)
    frozen = np.ones(n, dtype=bool)

    info_indices = np.argsort(metric)[:K]
    frozen[info_indices] = False

    return frozen


if __name__ == "__main__":
    metrics = qary_subchannels_monte_carlo(N, p, q, trials=50)
    frozen = freeze_bad_channels(metrics, K)

    print("Monte Carlo q-ary subchannel error estimates:")
    for i, err in enumerate(metrics):
        print(f"Subchannel {i}: {err:.4f}")

    print("Information symbol positions:", np.where(~frozen)[0])
    print("Frozen symbol positions:", np.where(frozen)[0])

    ranked = np.argsort(metrics)
    print("Reliability sequence best to worst:", *ranked)
