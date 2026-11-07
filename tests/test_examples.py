import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples" / "python"))

import bigram
import embeddings_demo
import toy_bpe


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
