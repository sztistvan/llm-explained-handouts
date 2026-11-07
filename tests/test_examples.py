import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples" / "python"))

import attention
import bigram
import embeddings_demo
import numpy as np
import toy_bpe


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
