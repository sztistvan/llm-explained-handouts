import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples" / "python"))

import moe
import rope
import sliding_window
import ssm


def test_moe_routes_exactly_k_experts_per_token():
    rng = np.random.default_rng(0)
    layer = moe.MoE(8, 16, n_experts=6, k=2, rng=rng)
    x = rng.normal(size=(50, 8))
    out, load = layer.forward(x)
    assert out.shape == x.shape
    assert load.sum() == 50 * 2


def test_moe_output_matches_manual_top_k_mixture():
    rng = np.random.default_rng(1)
    layer = moe.MoE(8, 16, n_experts=4, k=2, rng=rng)
    x = rng.normal(size=(1, 8))
    out, _ = layer.forward(x)
    logits = (x @ layer.router)[0]
    top = np.argsort(logits)[-2:]
    gates = moe.softmax(logits[top])
    expected = sum(g * layer._expert(x[0], *layer.experts[e]) for g, e in zip(gates, top))
    assert np.allclose(out[0], expected)


def test_moe_active_params_fraction():
    layer = moe.MoE(16, 64, n_experts=8, k=2, rng=np.random.default_rng(0))
    c = layer.param_counts()
    assert c["active_per_token"] < 0.3 * c["total"]
    dense = moe.MoE(16, 64, n_experts=8, k=8, rng=np.random.default_rng(0)).param_counts()
    assert dense["active_per_token"] == dense["total"]


def test_sliding_window_mask_properties():
    m = sliding_window.sliding_window_mask(10, 3)
    assert not np.triu(m, k=1).any()                 # still causal
    assert m.sum(axis=1).max() == 3                  # never more than `window` keys
    assert m[9].nonzero()[0].tolist() == [7, 8, 9]
    assert np.array_equal(sliding_window.sliding_window_mask(10, 10), sliding_window.causal_mask(10))


def test_receptive_field_grows_with_layers():
    m = sliding_window.sliding_window_mask(20, 4)
    for layers in (1, 2, 3):
        r = sliding_window.reach(m, layers)
        assert r[19].sum() == layers * (4 - 1) + 1   # L*(w-1)+1 positions


def test_window_attention_is_much_cheaper_for_long_sequences():
    assert sliding_window.attention_pairs(8000, 512) < 0.15 * sliding_window.attention_pairs(8000)


def test_ssm_recurrent_equals_convolution():
    rng = np.random.default_rng(0)
    A = rng.uniform(0.5, 0.95, 6)
    B, C = rng.normal(size=6), rng.normal(size=6)
    x = rng.normal(size=40)
    y1, h = ssm.ssm_recurrent(x, A, B, C)
    assert np.allclose(y1, ssm.ssm_convolution(x, A, B, C))
    assert h.size == 6                               # constant-size state


def test_rope_score_depends_only_on_relative_offset():
    rng = np.random.default_rng(0)
    q, k = rng.normal(size=16), rng.normal(size=16)
    a = rope.score(q, k, 10, 7)
    b = rope.score(q, k, 500, 497)
    assert a == pytest.approx(b)
    assert rope.score(q, k, 10, 7) != pytest.approx(rope.score(q, k, 10, 5))


def test_rope_preserves_vector_length():
    x = np.random.default_rng(1).normal(size=16)
    assert np.linalg.norm(rope.rope(x, 123)) == pytest.approx(np.linalg.norm(x))
