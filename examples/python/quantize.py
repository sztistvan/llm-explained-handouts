"""Weight quantization with per-group scales (handout 06).

Symmetric integer quantization: w ~= q * scale, q in [-(2^(b-1)-1), 2^(b-1)-1].
Smaller groups = a scale that fits each group better = lower error, slightly more memory.
"""
import numpy as np


def quantize(w, bits=8, group_size=None):
    """Return (q, scales). group_size=None -> one scale for the whole tensor."""
    flat = w.reshape(-1)
    g = group_size or flat.size
    assert flat.size % g == 0, "tensor size must be divisible by group size"
    groups = flat.reshape(-1, g)
    qmax = 2 ** (bits - 1) - 1
    scales = np.abs(groups).max(axis=1, keepdims=True) / qmax
    scales[scales == 0] = 1.0
    q = np.clip(np.round(groups / scales), -qmax, qmax).astype(np.int8)
    return q, scales


def dequantize(q, scales, shape):
    return (q * scales).reshape(shape)


def memory_bytes(n_params, bits, group_size=None, scale_bytes=2):
    n_groups = 1 if group_size is None else n_params / group_size
    return n_params * bits / 8 + n_groups * scale_bytes


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    w = rng.normal(0, 0.02, (1024, 1024))
    w[0, 0] = 0.5  # one outlier: typical of real weights, and bad for a single shared scale
    print(f"{'setting':<22}{'mean abs error':>16}{'bytes/param':>14}")
    for bits, g in [(8, None), (8, 64), (4, None), (4, 64), (4, 32)]:
        q, s = quantize(w, bits, g)
        err = np.abs(dequantize(q, s, w.shape) - w).mean()
        bpp = memory_bytes(w.size, bits, g) / w.size
        label = f"int{bits}, " + ("one scale" if g is None else f"group {g}")
        print(f"{label:<22}{err:>16.6f}{bpp:>14.3f}")
    print(f"{'float16 (baseline)':<22}{'-':>16}{2.0:>14.3f}")
