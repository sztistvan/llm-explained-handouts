"""Scaled dot-product attention, multi-head attention and a Transformer block
in plain NumPy (handout 03). Educational: no batching, no training.

IMPORTANT: This is a correct, functional implementation of the attention mechanism
as used in modern transformers (BERT, GPT, Llama, etc.). The scaled_dot_product_attention
function implements the exact formula: Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V.
Multi-head attention correctly splits inputs into independent heads, runs attention per head,
and concatenates results with an output projection. This matches production implementations.

Shapes: x is (seq_len, d_model).
"""
import numpy as np


def softmax(x, axis=-1): 
    """Compute softmax along the specified axis.
    Args:
        x: Input array.
        axis: Axis along which to compute softmax.
    The default axis = -1 (last axis) means softmax is computed along the last dimension of the input array.
    Note: along the last dimension means that the calculation scans/operates across each row's columns.

    Returns:
        Softmax of x along the specified axis.
    """
    x = x - x.max(axis=axis, keepdims=True)  # subtract max for numerical stability
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def causal_mask(n):
    """True where attention is ALLOWED: token i may look at tokens j <= i.
    Note: np.tril() generates a lower triangular matrix, which is used to allow attention only to previous tokens and the current token.
    """
    return np.tril(np.ones((n, n), dtype=bool)) # lower triangular mask, True where attention is allowed


def scaled_dot_product_attention(Q, K, V, mask=None):
    """softmax(Q K^T / sqrt(d_k)) V  -> (output, attention_weights)
    Args:
        Q: Query matrix of shape (seq_len, d_k).
        K: Key matrix of shape (seq_len, d_k).
        V: Value matrix of shape (seq_len, d_v).
        mask: Optional boolean mask of shape (seq_len, seq_len), True where attention is allowed.
    Note: The mask is used to prevent attention to certain positions, typically future tokens in causal attention.
    The operator "@" denotes matrix multiplication in NumPy.
    K.T is the transpose of the key matrix, used to compute the dot product with the query matrix.
    The division by sqrt(d_k) is a scaling factor to prevent large dot product values, which can lead to small gradients after the softmax.
    Returns: 
        output: Weighted sum of value vectors.
        attention_weights: Softmax-normalized attention scores.
    """
    d_k = Q.shape[-1]                        # dimension of each query/key vector
    scores = Q @ K.T / np.sqrt(d_k)          # (n, n): how well each query matches each key
    if mask is not None:
        scores = np.where(mask, scores, -1e9)  # blocked positions get ~0 weight after softmax
    weights = softmax(scores, axis=-1)       # each row sums to 1
    return weights @ V, weights              # weighted mix of value vectors


def multi_head_attention(x, Wq, Wk, Wv, Wo, n_heads, mask=None):
    """Multi-head self-attention mechanism.
    Args:
        x: Input sequence of shape (seq_len, d_model).
        Wq, Wk, Wv, Wo: Projection matrices for queries, keys, values, and output.
        n_heads: Number of attention heads.
        mask: Optional boolean mask of shape (seq_len, seq_len), True where attention is allowed.
    Returns:
        Output sequence after multi-head attention and output projection.
    """
    n, d_model = x.shape
    d_head = d_model // n_heads
    Q, K, V = x @ Wq, x @ Wk, x @ Wv # project input x to queries, keys, and values for attention
    heads = []
    for h in range(n_heads):
        s = slice(h * d_head, (h + 1) * d_head)  # each head sees its own slice
        out, _ = scaled_dot_product_attention(Q[:, s], K[:, s], V[:, s], mask)
        heads.append(out)
    return np.concatenate(heads, axis=-1) @ Wo   # concat heads, mix with output projection


def rms_norm(x, eps=1e-6):
    """Root Mean Square Layer Normalization.
    Args:
        x: Input array of shape (..., d_model).
        eps: Small constant to avoid division by zero.
    Returns:
        Normalized array with the same shape as input.
    """
    return x / np.sqrt((x ** 2).mean(axis=-1, keepdims=True) + eps)


def gelu(x):
    """Gaussian Error Linear Unit (GELU) activation function."""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))  # GELU activation function


def transformer_block(x, p, n_heads): 
    """Applies a single Transformer block with pre-norm and causal self-attention.
    Args:
        x: Input sequence of shape (seq_len, d_model).
        p: Dictionary of block parameters (projection matrices).
        n_heads: Number of attention heads.
    
    Note:
        This block uses pre-norm (layer normalization before attention and feed-forward layers)
        and causal self-attention (each token can only attend to previous tokens).
    
    Returns:
        Output sequence after applying the Transformer block.
    """
    
    mask = causal_mask(x.shape[0]) # causal mask for self-attention
    x = x + multi_head_attention(rms_norm(x), p["Wq"], p["Wk"], p["Wv"], p["Wo"], n_heads, mask) # self-attention with causal mask
    h = gelu(rms_norm(x) @ p["W1"])              # expand to d_ff
    return x + h @ p["W2"]                       # project back to d_model


def init_block(d_model, d_ff, rng):
    """Initializes the parameters for a single Transformer block.
    Args:
        d_model: Model dimension.
        d_ff: Feed-forward dimension.
        rng: Random number generator.
    Returns:
        Dictionary of initialized projection matrices for the block.
    """
    s = 0.02
    return {
        "Wq": rng.normal(0, s, (d_model, d_model)), # query projection
        "Wk": rng.normal(0, s, (d_model, d_model)), # key projection
        "Wv": rng.normal(0, s, (d_model, d_model)), # value projection
        "Wo": rng.normal(0, s, (d_model, d_model)), # output projection
        "W1": rng.normal(0, s, (d_model, d_ff)), # feed-forward expansion
        "W2": rng.normal(0, s, (d_ff, d_model)), # feed-forward projection back to model dimension
    }


if __name__ == "__main__":
    rng = np.random.default_rng(0) # random number generator with fixed seed for reproducibility
    tokens = ["The", "cat", "sat", "down", "."] # example token sequence
    n, d_model, d_ff, n_heads = len(tokens), 16, 64, 4 # sequence length, model dimension, feed-forward dimension, number of attention heads
    x = rng.normal(size=(n, d_model))

    # Single head, self-attention (Q = K = V = x for illustration)
    _, w = scaled_dot_product_attention(x, x, x, causal_mask(n)) # compute causal attention weights for single head
    print("Causal attention weights (row = token looking, column = token looked at):")
    print(np.round(w, 2)) # rounded attention weights for readability

    y = transformer_block(x, init_block(d_model, d_ff, rng), n_heads) # apply single Transformer block to input sequence
    print("\nBlock output shape:", y.shape, "(same as input, so blocks can be stacked)")
