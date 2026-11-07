"""Rotary position embedding (RoPE) and position interpolation (handout 07, builds on 02).

RoPE rotates each (even, odd) pair of a query/key vector by an angle proportional to the
token position. The dot product q.k then depends only on the RELATIVE distance.
"""
import numpy as np


def rope(x, pos, base=10000.0):
    d = len(x)
    theta = base ** (-np.arange(0, d, 2) / d)   # one frequency per pair
    ang = pos * theta
    cos, sin = np.cos(ang), np.sin(ang)
    out = np.empty_like(x)
    out[0::2] = x[0::2] * cos - x[1::2] * sin
    out[1::2] = x[0::2] * sin + x[1::2] * cos
    return out


def score(q, k, m, n, scale=1.0):
    """Attention score between query at position m and key at position n.
    scale < 1 implements position interpolation (squeeze positions into the trained range)."""
    return rope(q, m * scale) @ rope(k, n * scale)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    q, k = rng.normal(size=16), rng.normal(size=16)
    print("same offset (3), different absolute positions:")
    for m in (3, 50, 1000):
        print(f"  positions ({m}, {m - 3}): score = {score(q, k, m, m - 3):.6f}")
    print("\nposition interpolation: a 4x longer text is squeezed into the trained range")
    trained, new = 4096, 16384
    s = trained / new
    print(f"  position {new - 1} is treated like position {(new - 1) * s:.1f} (scale {s})")
