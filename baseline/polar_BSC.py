import numpy as np

N = 256
p = 0.05
K = 256

# bit reversal ordering
def polar_transform(u):
    if len(u) == 1:
        return u
    else:
        x = np.zeros(len(u), dtype=np.int64)
        x[:len(u)//2] = polar_transform((u[::2] + u[1::2]) % 2)
        x[len(u)//2:] = polar_transform(u[1::2])
        return x


#channel
def bsc(x, p, rng=None):
    x = np.asarray(x, dtype=np.int64)
    if rng is None:
        r = np.random.random_sample(len(x))
    else:
        r = rng.random(len(x))

    y = np.array(x, copy=True, dtype=np.int64)
    y[r < p] ^= 1
    return y


def freeze_bad_channels(metric, K):
    n = len(metric)
    frozen = np.ones(n, dtype=bool)
    info_indices = np.argsort(metric)[:K]
    frozen[info_indices] = False
    return frozen

def f(a, b):
    return (1.0 + a * b) / (a + b)

def g(a, b, u):
    if u == 0:
        return a * b
    return b / a

#convert BSC observations to likelihood ratios.
#LR = P(y | x=0) / P(y | x=1)
def bsc_obs_to_lr(y, p):
    y = np.asarray(y, dtype=np.int64)
    lr = np.empty(len(y), dtype=np.float64)

    if p == 0.0:
        for i in range(len(y)):
            lr[i] = np.inf if y[i] == 0 else 0.0
        return lr

    lr_y0 = (1.0 - p) / p
    lr_y1 = p / (1.0 - p)

    for i in range(len(y)):
        lr[i] = lr_y0 if y[i] == 0 else lr_y1

    return lr


#LR-based SC decoder
def sc_decode_lr(lr_vec, frozen):
    def _decode_node(obs, frozen_node):
        N = len(obs)

        if N == 1:
            if frozen_node[0]:
                bit = 0
            else:
                bit = 0 if obs[0] >= 1.0 else 1

            u_hat_node = np.array([bit], dtype=np.int64)
            x_hat_node = np.array([bit], dtype=np.int64)
            return u_hat_node, x_hat_node

        half = N // 2

        upper_obs = np.zeros(half, dtype=np.float64)
        for i in range(half):
            upper_obs[i] = f(obs[2 * i], obs[2 * i + 1])

        u_hat_upper, x_hat_upper = _decode_node(upper_obs, frozen_node[:half])

        lower_obs = np.zeros(half, dtype=np.float64)
        for i in range(half):
            lower_obs[i] = g(obs[2 * i], obs[2 * i + 1], x_hat_upper[i])

        u_hat_lower, x_hat_lower = _decode_node(lower_obs, frozen_node[half:])

        u_hat_node = np.zeros(N, dtype=np.int64)
        u_hat_node[:half] = u_hat_upper
        u_hat_node[half:] = u_hat_lower

        x_hat_node = np.zeros(N, dtype=np.int64)
        for i in range(half):
            x_hat_node[2 * i] = (x_hat_upper[i] + x_hat_lower[i]) % 2
            x_hat_node[2 * i + 1] = x_hat_lower[i]

        return u_hat_node, x_hat_node

    u_hat, _ = _decode_node(np.asarray(lr_vec, dtype=np.float64),
                            np.asarray(frozen, dtype=bool))
    return u_hat


#compute LR for bit-channel i under SC, assuming all previous bits are known.
def sc_bit_lr(obs, u_prev, bit_index):
    obs = np.asarray(obs, dtype=np.float64)
    u_prev = np.asarray(u_prev, dtype=np.int64)

    def _bit_lr(obs_node, prev_node, target):
        n = len(obs_node)

        if n == 1:
            return obs_node[0]

        half = n // 2
        upper_obs = np.zeros(half, dtype=np.float64)
        for i in range(half):
            upper_obs[i] = f(obs_node[2 * i], obs_node[2 * i + 1])

        if target < half:
            return _bit_lr(upper_obs, prev_node, target)

        u_upper = prev_node[:half]
        x_upper = polar_transform(u_upper)

        lower_obs = np.zeros(half, dtype=np.float64)
        for i in range(half):
            lower_obs[i] = g(obs_node[2 * i], obs_node[2 * i + 1], x_upper[i])

        return _bit_lr(lower_obs, prev_node[half:], target - half)

    return _bit_lr(obs, u_prev, bit_index)


#monte carlo for BSC
#for each bit-channel, transmit the all-zero codeword, assume previous bits are known, and estimate the SC decision error probability for bit i
# def bsc_subchannels_monte_carlo(N, p, trials, seed=None):
#     rng = np.random.default_rng(seed)
#     x_zero = np.zeros(N, dtype=np.int64)
#     error_probs = np.zeros(N, dtype=np.float64)

#     for bit_index in range(N):
#         errors = 0
#         u_prev = np.zeros(bit_index, dtype=np.int64)

#         for _ in range(trials):
#             y = bsc(x_zero, p, rng=rng)
#             lr_vec = bsc_obs_to_lr(y, p)
#             lr_bit = sc_bit_lr(lr_vec, u_prev, bit_index)

#             # All-zero transmission + SC rule: decide 1 only if LR < 1.
#             if lr_bit < 1.0:
#                 errors += 1

#         error_probs[bit_index] = errors / trials

#     return error_probs

def bsc_subchannels_monte_carlo(N, p, trials, seed=None):
    rng = np.random.default_rng(seed)
    error_probs = np.zeros(N, dtype=np.float64)

    for bit_index in range(N):
        errors = 0

        for _ in range(trials):
            # Random full u vector, not all-zero
            u = rng.integers(0, 2, N, dtype=np.int64)

            x = polar_transform(u)
            y = bsc(x, p, rng=rng)
            lr_vec = bsc_obs_to_lr(y, p)

            # Genie gives previous true bits
            lr_bit = sc_bit_lr(lr_vec, u[:bit_index], bit_index)

            u_decision = 0 if lr_bit >= 1.0 else 1

            if u_decision != u[bit_index]:
                errors += 1

        error_probs[bit_index] = errors / trials

    return error_probs


if __name__ == "__main__":
    metrics = bsc_subchannels_monte_carlo(N, p, trials=500)
    frozen = freeze_bad_channels(metrics, K)

    u = np.zeros(N, dtype=np.int64)
    u[~frozen] = np.random.randint(0, 2, K)

    x = polar_transform(u)
    y = bsc(x, p)
    lr_vec = bsc_obs_to_lr(y, p)
    u_hat = sc_decode_lr(lr_vec, frozen)

    print("Monte Carlo bit-channel error estimates =", metrics)
    print("Information bit positions (0-based):", np.where(~frozen)[0])
    print("u     =", u)
    print("u_hat =", u_hat)

    ranked = np.argsort(metrics)  # +1 for 1-based indexing like the slide
    print("Reliability sequence (best to worst):", *ranked)