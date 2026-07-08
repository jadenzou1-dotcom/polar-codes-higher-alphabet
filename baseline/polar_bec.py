import numpy as np

N= 64
e = 0.01
K= 8

# bit reversal ordering
def polar_transform(u):
    if len(u) == 1:
        return u
    else:
        x = np.zeros(len(u), dtype=np.int64)
        x[:len(u)//2] = polar_transform((u[::2] + u[1::2]) % 2)
        x[len(u)//2:] = polar_transform(u[1::2])
        return x

#BEC channel model
def bec(x, e):

    r = np.random.random_sample(len(x))
    #print("random =", r)

    y = np.zeros(len(x), dtype=np.float64)
    for i in range(len(x)):
        if (r[i]<e):
            y[i] = 0.5
        else: 
            y[i] = x[i]

    return y

#Determining good subchannels for BEC
def bec_subchannels(N, e):

    Z = np.array([e], dtype=np.float64)

    while len(Z) < N:
        Z_new = np.zeros(2 * len(Z), dtype=np.float64)
        for i in range(len(Z)):
            Z_new[2*i]     = 2*Z[i] - Z[i]**2 # bad channel
            Z_new[2*i + 1] = Z[i]**2 # good channel
        Z = Z_new

    return Z

def freeze_bad_channels(Z, K):

    n = len(Z)
    frozen = np.ones(n, dtype=bool)
    info_indices = np.argsort(Z)[:K]
    frozen[info_indices] = False

    # frozen = np.ones(N, dtype=bool) #start with all bits frozen
    # info_indices = np.argsort(Z)[:K] #info_indicies will have K number of smallest Zs in Z array
    # frozen[info_indices] = False #makes the best "info_indicies" subchannels unfrozen

    return frozen   

def f(a, b):
    """
    equatino 75, Arikan LR upper combine:

    Inputs a,b are likelihood ratios in [0, inf].
    
    0  = certain 1
    inf = certain 0
    1  = erasure

    """
    if np.isinf(a) and np.isinf(b):
        return np.inf
    if a == 0.0 and b == 0.0:
        return np.inf
    if (np.isinf(a) and b == 0.0) or (a == 0.0 and np.isinf(b)):
        return 0.0

    denom = a + b
    num = 1.0 + a * b

    # extra guards for inf arithmetic
    if a == 1.0 or b == 1.0:
        return 1.0

    return num / denom


def g(a, b, u):
    
    #equation 76, Arikan LR lower combine:
    if u == 0:
        if (np.isinf(a) and b == 0.0) or (a == 0.0 and np.isinf(b)):
            return 1.0   # treat as erasure

        return b * a

    else:
        if a == 0.0:
            return np.inf   # b / 0 → ∞ (certain 0)
        if np.isinf(a) and np.isinf(b):
            return 1.0      # ∞ / ∞ → erasure

        return b / a


def bec_obs_to_lr(y):
    """
    Convert BEC outputs to likelihood ratios:

        y = 0   -> LR = inf   (definitely x=0)
        y = 1   -> LR = 0     (definitely x=1)
        y = 0.5 -> LR = 1     (erasure, equal likelihood)

    """
    y = np.asarray(y, dtype=np.float64)
    lr = np.empty_like(y, dtype=np.float64)

    for i in range(len(y)):
        if y[i] == 0.0:
            lr[i] = np.inf
        elif y[i] == 1.0:
            lr[i] = 0.0
        else:   # erasure
            lr[i] = 1.0

    return lr


def sc_decode_lr(lr_vec, frozen):

    def _decode_node(obs, frozen_node):
        N = len(obs)

        # Base case
        if N == 1:
            if frozen_node[0]:
                bit = 0
            else:
                # Arikan decision rule: choose 0 if LR >= 1, else 1
                bit = 0 if obs[0] >= 1.0 else 1

            u_hat_node = np.array([bit], dtype=np.int64)
            x_hat_node = np.array([bit], dtype=np.int64)
            return u_hat_node, x_hat_node

        half = N // 2

        # Upper recursion
        upper_obs = np.zeros(half, dtype=np.float64)
        for i in range(half):
            upper_obs[i] = f(obs[2 * i], obs[2 * i + 1])

        u_hat_upper, x_hat_upper = _decode_node(upper_obs, frozen_node[:half])

        # Lower recursion
        lower_obs = np.zeros(half, dtype=np.float64)
        for i in range(half):
            lower_obs[i] = g(obs[2 * i], obs[2 * i + 1], x_hat_upper[i])

        u_hat_lower, x_hat_lower = _decode_node(lower_obs, frozen_node[half:])

        # Combine u-hat in input order
        u_hat_node = np.zeros(N, dtype=np.int64)
        u_hat_node[:half] = u_hat_upper
        u_hat_node[half:] = u_hat_lower

        # Reconstruct hard x-hat to feed parent g() calls
        x_hat_node = np.zeros(N, dtype=np.int64)
        for i in range(half):
            x_hat_node[2 * i] = (x_hat_upper[i] + x_hat_lower[i]) % 2
            x_hat_node[2 * i + 1] = x_hat_lower[i]

        return u_hat_node, x_hat_node

    u_hat, _ = _decode_node(np.asarray(lr_vec, dtype=np.float64),
                            np.asarray(frozen, dtype=bool))
    return u_hat


Z = bec_subchannels(N, e) #calculate Bhattacharyya parameters for all N subchannels
frozen = freeze_bad_channels(Z, K) #sort and freeze the worst N-K subchannels
# u = np.zeros(N, dtype=np.int64) #intialize u vector of length N with all 0s
# u[~frozen] = np.random.randint(0, 2, K) #assign random info bits to unfrozen positions
# x = polar_transform(u)
# y = bec(x, e)
# lr_vec = bec_obs_to_lr(y)
# u_hat = sc_decode_lr(lr_vec, frozen)


print("Z =", Z)
# #print("frozen =", frozen)
#print("u =    ", u)
# #print("x =", x)
# #print("y =", y)
#print("u_hat =", u_hat)
# info_indices = np.where(~frozen)[0]
# print("Information bit positions (0-based):" , info_indices)

#debug print out each recursion