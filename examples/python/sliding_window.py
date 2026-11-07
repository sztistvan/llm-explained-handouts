"""Sliding-window attention masks and how far information can travel (handout 07)."""
import numpy as np


def causal_mask(n):
    return np.tril(np.ones((n, n), dtype=bool))


def sliding_window_mask(n, window):
    """True where attention is allowed: token i sees tokens j with i-window < j <= i."""
    i = np.arange(n)[:, None]
    j = np.arange(n)[None, :]
    return (j <= i) & (j > i - window)


def reach(mask, layers):
    """Which positions can influence each token after `layers` stacked layers (boolean matrix power)."""
    m = mask.astype(int)
    r = np.eye(len(mask), dtype=int)
    for _ in range(layers):
        r = ((r @ m) > 0).astype(int)
    return r.astype(bool)


def attention_pairs(n, window=None):
    return int((causal_mask(n) if window is None else sliding_window_mask(n, window)).sum())


if __name__ == "__main__":
    n, w = 16, 4
    print(f"window={w}: token 15 directly sees positions", np.where(sliding_window_mask(n, w)[15])[0].tolist())
    for layers in (1, 2, 3):
        r = reach(sliding_window_mask(n, w), layers)
        print(f"after {layers} layer(s): token 15 is influenced by positions {np.where(r[15])[0].tolist()}")
    for n in (1_000, 8_000):
        print(f"n={n:>5}: full causal pairs {attention_pairs(n):>10,} | window 512 pairs {attention_pairs(n, 512):>10,}")
