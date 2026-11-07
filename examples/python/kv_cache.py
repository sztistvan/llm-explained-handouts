"""KV cache: correctness, saved work, and memory sizing (handout 06).

Single-head causal attention, decoding one token at a time.
"""
import numpy as np


def softmax(x):
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


class Head:
    def __init__(self, d_model, d_k, rng):
        s = 0.3
        self.Wq = rng.normal(0, s, (d_model, d_k))
        self.Wk = rng.normal(0, s, (d_model, d_k))
        self.Wv = rng.normal(0, s, (d_model, d_k))
        self.d_k = d_k
        self.k_projections = 0  # counts how many K/V projections were computed (the "work")

    def attend_no_cache(self, xs):
        """Recompute K and V for ALL tokens, return output for the last one."""
        K, V = xs @ self.Wk, xs @ self.Wv
        self.k_projections += len(xs)
        q = xs[-1] @ self.Wq
        w = softmax(K @ q / np.sqrt(self.d_k))
        return w @ V

    def attend_cached(self, x_new, cache):
        """Compute K and V only for the NEW token, append to the cache, reuse the rest."""
        cache["K"].append(x_new @ self.Wk)
        cache["V"].append(x_new @ self.Wv)
        self.k_projections += 1
        K, V = np.array(cache["K"]), np.array(cache["V"])
        q = x_new @ self.Wq
        w = softmax(K @ q / np.sqrt(self.d_k))
        return w @ V


def kv_cache_bytes(layers, kv_heads, head_dim, seq_len, batch=1, bytes_per_value=2):
    """2 (K and V) x layers x kv_heads x head_dim x tokens x batch x bytes."""
    return 2 * layers * kv_heads * head_dim * seq_len * batch * bytes_per_value


def weights_bytes(params, bytes_per_param=2):
    return params * bytes_per_param


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    head = Head(d_model=16, d_k=8, rng=rng)
    xs = rng.normal(size=(50, 16))

    out_full = [head.attend_no_cache(xs[: t + 1]) for t in range(50)]
    work_full = head.k_projections
    head.k_projections = 0
    cache = {"K": [], "V": []}
    out_cached = [head.attend_cached(xs[t], cache) for t in range(50)]
    print("identical outputs:", np.allclose(out_full, out_cached))
    print(f"K/V projections for 50 tokens: no cache = {work_full}, with cache = {head.k_projections}")

    # Memory sizing for a Llama-3-8B-like configuration (32 layers, 8 KV heads, head_dim 128, 16-bit)
    per_token = kv_cache_bytes(32, 8, 128, 1)
    print(f"\nKV cache per token: {per_token / 1024:.0f} KiB")
    for ctx in (8_192, 128_000):
        gib = kv_cache_bytes(32, 8, 128, ctx) / 2**30
        print(f"  context {ctx:>7,}: {gib:6.1f} GiB per sequence")
    print(f"weights of an 8B model at 16-bit: {weights_bytes(8e9) / 2**30:.1f} GiB")
