"""Scaled dot-product attention, multi-head attention and a Transformer block
in plain NumPy (handout 03). Educational: no batching, no training.

Shapes: x is (seq_len, d_model).
"""
import numpy as np


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)  # subtract max for numerical stability
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def causal_mask(n):
    """True where attention is ALLOWED: token i may look at tokens j <= i."""
    return np.tril(np.ones((n, n), dtype=bool))


def scaled_dot_product_attention(Q, K, V, mask=None):
    """softmax(Q K^T / sqrt(d_k)) V  -> (output, attention_weights)"""
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)          # (n, n): how well each query matches each key
    if mask is not None:
        scores = np.where(mask, scores, -1e9)  # blocked positions get ~0 weight after softmax
    weights = softmax(scores, axis=-1)       # each row sums to 1
    return weights @ V, weights              # weighted mix of value vectors


def multi_head_attention(x, Wq, Wk, Wv, Wo, n_heads, mask=None):
    n, d_model = x.shape
    d_head = d_model // n_heads
    Q, K, V = x @ Wq, x @ Wk, x @ Wv
    heads = []
    for h in range(n_heads):
        s = slice(h * d_head, (h + 1) * d_head)  # each head sees its own slice
        out, _ = scaled_dot_product_attention(Q[:, s], K[:, s], V[:, s], mask)
        heads.append(out)
    return np.concatenate(heads, axis=-1) @ Wo   # concat heads, mix with output projection


def rms_norm(x, eps=1e-6):
    return x / np.sqrt((x ** 2).mean(axis=-1, keepdims=True) + eps)


def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


def transformer_block(x, p, n_heads):
    """Pre-norm decoder block: x + Attn(norm(x)), then x + MLP(norm(x))."""
    mask = causal_mask(x.shape[0])
    x = x + multi_head_attention(rms_norm(x), p["Wq"], p["Wk"], p["Wv"], p["Wo"], n_heads, mask)
    h = gelu(rms_norm(x) @ p["W1"])              # expand to d_ff
    return x + h @ p["W2"]                       # project back to d_model


def init_block(d_model, d_ff, rng):
    s = 0.02
    return {
        "Wq": rng.normal(0, s, (d_model, d_model)),
        "Wk": rng.normal(0, s, (d_model, d_model)),
        "Wv": rng.normal(0, s, (d_model, d_model)),
        "Wo": rng.normal(0, s, (d_model, d_model)),
        "W1": rng.normal(0, s, (d_model, d_ff)),
        "W2": rng.normal(0, s, (d_ff, d_model)),
    }


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    tokens = ["The", "cat", "sat", "down"]
    n, d_model, d_ff, n_heads = len(tokens), 16, 64, 4
    x = rng.normal(size=(n, d_model))

    # Single head, self-attention (Q = K = V = x for illustration)
    _, w = scaled_dot_product_attention(x, x, x, causal_mask(n))
    print("Causal attention weights (row = token looking, column = token looked at):")
    print(np.round(w, 2))

    y = transformer_block(x, init_block(d_model, d_ff, rng), n_heads)
    print("\nBlock output shape:", y.shape, "(same as input, so blocks can be stacked)")
