import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples" / "python"))

import attention
import bigram
import embeddings_demo
import numpy as np
import lora
import sampling
import toy_bpe
import train_bigram


def test_training_reduces_loss_and_learns_counts():
    words = "the cat sat on the mat the cat ate the fish".split()
    vocab, ids = train_bigram.encode(words)
    W, losses = train_bigram.train(ids, len(vocab), steps=500)
    assert losses[-1] < losses[0] * 0.5
    assert all(a >= b - 1e-9 for a, b in zip(losses, losses[1:]))   # full-batch GD on convex loss
    probs = train_bigram.softmax_rows(W[[vocab.index("the")]])[0]
    # "the" is followed by cat (2x), mat, fish -> learned ~ counts 0.5 / 0.25 / 0.25 (+ ate unseen)
    assert probs[vocab.index("cat")] == pytest.approx(0.5, abs=0.05)
    assert probs[vocab.index("mat")] == pytest.approx(0.25, abs=0.05)


def test_lora_starts_identical_to_base_and_merges_exactly():
    rng = np.random.default_rng(0)
    layer = lora.LoRALinear(rng.normal(size=(32, 16)), r=4, alpha=8, rng=rng)
    x = rng.normal(size=(5, 16))
    assert np.allclose(layer.forward(x), x @ layer.W0.T)            # B = 0 => no change at start
    layer.B = rng.normal(size=layer.B.shape)                        # pretend training happened
    assert np.allclose(layer.forward(x), x @ layer.merged_weight().T)


def test_lora_trainable_params_far_fewer():
    rng = np.random.default_rng(0)
    layer = lora.LoRALinear(rng.normal(size=(1024, 1024)), r=8, alpha=16, rng=rng)
    assert layer.trainable_params() == 8 * 1024 * 2
    assert layer.trainable_params() < 0.02 * layer.W0.size


LOGITS = np.array([3.0, 2.0, 0.5, -1.0])


def test_temperature_sharpens_and_flattens():
    cold = sampling.next_token_distribution(LOGITS, temperature=0.2)
    base = sampling.next_token_distribution(LOGITS, temperature=1.0)
    hot = sampling.next_token_distribution(LOGITS, temperature=2.0)
    assert cold.max() > base.max() > hot.max()
    for p in (cold, base, hot):
        assert p.sum() == pytest.approx(1.0)


def test_temperature_zero_is_greedy():
    assert sampling.sample(LOGITS, temperature=0) == 0


def test_top_k_keeps_k_tokens():
    p = sampling.next_token_distribution(LOGITS, top_k=2)
    assert (p > 0).sum() == 2 and p.sum() == pytest.approx(1.0)


def test_top_p_keeps_smallest_set_reaching_p():
    probs = sampling.softmax(LOGITS)          # ~[0.68, 0.25, 0.06, 0.01]
    p = sampling.top_p_filter(probs, 0.9)
    assert (p > 0).sum() == 2                 # 0.68 + 0.25 >= 0.9
    p = sampling.top_p_filter(probs, 0.5)
    assert (p > 0).sum() == 1                 # top token alone already >= 0.5


def test_sampling_never_picks_filtered_tokens():
    rng = np.random.default_rng(0)
    picks = {sampling.sample(LOGITS, 1.0, top_k=2, rng=rng) for _ in range(200)}
    assert picks <= {0, 1}


def test_beam_search_beats_greedy_on_toy_model():
    g_seq, g_lp = sampling.greedy_search(sampling.toy_model, 2)
    best_seq, best_lp = sampling.beam_search(sampling.toy_model, 2, 2)[0]
    assert g_seq[0] == "A" and best_seq == ["B", "z"]
    assert best_lp > g_lp


def test_softmax_rows_sum_to_one():
    s = attention.softmax(np.random.default_rng(0).normal(size=(4, 5)))
    assert np.allclose(s.sum(axis=-1), 1.0)


def test_causal_attention_ignores_future():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(5, 8))
    _, w = attention.scaled_dot_product_attention(x, x, x, attention.causal_mask(5))
    assert np.allclose(np.triu(w, k=1), 0)          # nothing above the diagonal
    assert np.allclose(w.sum(axis=-1), 1.0)
    assert w[0, 0] == pytest.approx(1.0)            # first token can only see itself


def test_causal_output_independent_of_future_tokens():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(5, 8))
    out1, _ = attention.scaled_dot_product_attention(x, x, x, attention.causal_mask(5))
    x2 = x.copy()
    x2[4] = rng.normal(size=8)                      # change only the last token
    out2, _ = attention.scaled_dot_product_attention(x2, x2, x2, attention.causal_mask(5))
    assert np.allclose(out1[:4], out2[:4])          # earlier positions unaffected


def test_swapping_q_and_k_transposes_the_score_table():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(5, 8))
    Wq, Wk = rng.normal(size=(8, 4)), rng.normal(size=(8, 4))
    S = (X @ Wq) @ (X @ Wk).T                      # score_ij = q_i . k_j
    S_swapped = (X @ Wk) @ (X @ Wq).T
    assert np.allclose(S_swapped, S.T)             # swapped roles = reversed relation
    assert np.allclose(S, X @ (Wq @ Wk.T) @ X.T)   # only M = Wq Wk^T matters
    assert not np.allclose(S, S.T)                 # separate Wq, Wk give an asymmetric relation
    W = Wq
    S_same = (X @ W) @ (X @ W).T
    assert np.allclose(S_same, S_same.T)           # Wq == Wk would force a symmetric relation


def test_cached_keys_and_values_do_not_change_when_tokens_are_appended():
    rng = np.random.default_rng(5)
    Wk, Wv = rng.normal(size=(8, 4)), rng.normal(size=(8, 4))
    x = rng.normal(size=(6, 8))
    # Layer-0 style check: k_i, v_i depend only on token i's own vector
    assert np.allclose((x[:4] @ Wk), (x @ Wk)[:4])
    assert np.allclose((x[:4] @ Wv), (x @ Wv)[:4])
    # In a causal stack, a position's hidden state also ignores later tokens
    block = attention.init_block(8, 16, rng)
    h_short = attention.transformer_block(x[:4], block, n_heads=2)
    h_long = attention.transformer_block(x, block, n_heads=2)[:4]
    assert np.allclose(h_short, h_long)


def test_last_output_row_needs_only_the_last_query():
    """Row i of the attention output depends on query i only (plus keys and values)."""
    rng = np.random.default_rng(7)
    n, d = 6, 8
    X = rng.normal(size=(n, d))
    Wq, Wk, Wv = (rng.normal(size=(d, d)) for _ in range(3))
    Q, K, V = X @ Wq, X @ Wk, X @ Wv
    mask = attention.causal_mask(n)
    full, _ = attention.scaled_dot_product_attention(Q, K, V, mask)
    last, _ = attention.scaled_dot_product_attention(Q[-1:], K, V)       # last query row only
    assert np.allclose(full[-1], last[0])

    Q_changed = Q.copy()
    Q_changed[2] = rng.normal(size=d)                                    # change q_3 only
    changed, _ = attention.scaled_dot_product_attention(Q_changed, K, V, mask)
    assert [i for i in range(n) if not np.allclose(full[i], changed[i])] == [2]   # only output row 3 moves


def test_transformer_block_preserves_shape():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(6, 16))
    y = attention.transformer_block(x, attention.init_block(16, 64, rng), n_heads=4)
    assert y.shape == x.shape and np.isfinite(y).all()


def test_bpe_merges_and_unseen_word():
    merges = toy_bpe.train_bpe("low lower lowest newer newest wider widest low low", 10)
    assert merges[0] == (("l", "o"), "lo")
    assert toy_bpe.encode("lowest", merges) == ["low", "est"]
    assert toy_bpe.encode("slowest", merges) == ["s", "low", "est"]


def test_bpe_roundtrip():
    merges = toy_bpe.train_bpe("abababab cdcd", 5)
    assert "".join(toy_bpe.encode("ababcd", merges)) == "ababcd"


def test_bigram_generation_uses_only_seen_transitions():
    words = "the cat sat on the mat the cat ate the fish".split()
    counts = bigram.train(words)
    out = bigram.generate(counts, "the", 6, random.Random(0))
    for a, b in zip(out, out[1:]):
        assert b in counts[a]


def test_cosine_values_match_handout():
    e = embeddings_demo.emb
    assert embeddings_demo.cosine(e["king"], e["queen"]) == pytest.approx(0.66, abs=0.01)
    assert embeddings_demo.cosine(e["king"], e["apple"]) == pytest.approx(-0.44, abs=0.01)


def test_tiktoken_roundtrip():
    tiktoken = pytest.importorskip("tiktoken")
    try:
        enc = tiktoken.get_encoding("cl100k_base")
    except Exception as e:  # offline: encoding file cannot be downloaded
        pytest.skip(f"tiktoken encoding unavailable: {e}")
    text = "Tokenization is fun!"
    assert enc.decode(enc.encode(text)) == text
