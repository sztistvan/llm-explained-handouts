"""A minimal (non-selective) state-space layer (handout 07).

State: h_t = A * h_{t-1} + B * x_t     (A, B: vectors of length N, diagonal dynamics)
Output: y_t = C . h_t

Two equivalent views:
  * recurrent  : process tokens one by one with a FIXED-size state (cheap generation)
  * convolution: y = causal convolution of x with kernel K_j = C . (A^j * B) (parallel training)
Mamba makes A, B, C depend on the input ("selective" SSM); this fixed version is the simplest case.
"""
import numpy as np


def ssm_recurrent(x, A, B, C):
    h = np.zeros_like(A)
    ys = []
    for xt in x:
        h = A * h + B * xt
        ys.append(C @ h)
    return np.array(ys), h


def ssm_kernel(A, B, C, length):
    return np.array([C @ (A ** j * B) for j in range(length)])


def ssm_convolution(x, A, B, C):
    K = ssm_kernel(A, B, C, len(x))
    return np.array([sum(K[j] * x[t - j] for j in range(t + 1)) for t in range(len(x))])


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    N = 8
    A = rng.uniform(0.5, 0.95, N)   # |A| < 1 => old information fades smoothly
    B, C = rng.normal(size=N), rng.normal(size=N)
    x = rng.normal(size=64)
    y_rec, h = ssm_recurrent(x, A, B, C)
    y_conv = ssm_convolution(x, A, B, C)
    print("recurrent == convolution:", np.allclose(y_rec, y_conv))
    print(f"state size after 64 tokens: {h.size} numbers (an attention KV cache would hold 64 keys and values)")
    for n in (1_000, 100_000):
        print(f"after {n:>7,} tokens: SSM state = {N} numbers; KV cache grows to {n} entries per layer")
