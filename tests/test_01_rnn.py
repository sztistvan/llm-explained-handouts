import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples" / "python"))

import rnn_cell


def make_cell(seed=0):
    return rnn_cell.RNNCell(d_in=8, d_h=16, rng=np.random.default_rng(seed))


def test_hidden_state_size_is_constant_for_any_sequence_length():
    cell = make_cell()
    rng = np.random.default_rng(1)
    for n in (1, 10, 500):
        states = cell.run(rng.normal(size=(n, 8)))
        assert len(states) == n and all(s.shape == (16,) for s in states)


def test_unrolled_loop_matches_manual_steps_with_the_same_weights():
    cell = make_cell()
    xs = np.random.default_rng(2).normal(size=(4, 8))
    h = np.zeros(16)
    for x in xs:
        h = cell.step(x, h)                       # same cell, same weights, every step
    assert np.allclose(cell.run(xs)[-1], h)


def test_state_depends_on_the_previous_state():
    cell = make_cell()
    x = np.random.default_rng(3).normal(size=8)
    assert not np.allclose(cell.step(x, np.zeros(16)), cell.step(x, np.ones(16)))


def test_word_order_matters():
    cell = make_cell()
    xs = np.random.default_rng(4).normal(size=(6, 8))
    swapped = xs.copy()
    swapped[[0, 1]] = swapped[[1, 0]]
    assert not np.allclose(cell.run(xs)[-1], cell.run(swapped)[-1])


def test_states_stay_bounded_by_tanh():
    cell = make_cell()
    xs = 50 * np.random.default_rng(5).normal(size=(30, 8))   # huge inputs
    assert all(np.abs(s).max() <= 1.0 for s in cell.run(xs))


def test_with_small_inputs_the_fade_tracks_the_recurrent_scale_closely():
    """The doc's demo: tanh stays almost linear, so the influence is close to 0.9^k (but never above it)."""
    rng = np.random.default_rng(0)
    cell = rnn_cell.RNNCell(d_in=8, d_h=16, rng=rng)
    tokens = rng.normal(size=(60, 8))
    small = 0.02 * tokens
    x0_a, x0_b = 0.02 * rng.normal(size=8), 0.02 * rng.normal(size=8)
    base = rnn_cell.first_token_influence(cell, x0_a, x0_b, [])
    for length in (5, 10, 20, 50):
        bound = 0.9 ** (length - 1) * base
        infl = rnn_cell.first_token_influence(cell, x0_a, x0_b, small[: length - 1])
        assert 0.85 * bound <= infl <= bound + 1e-12


def test_influence_of_the_first_token_fades_at_least_as_fast_as_the_recurrent_scale():
    cell = make_cell()
    rng = np.random.default_rng(6)
    x0_a, x0_b = rng.normal(size=8), rng.normal(size=8)
    rest = rng.normal(size=(60, 8))
    base = rnn_cell.first_token_influence(cell, x0_a, x0_b, [])
    assert base > 0.1
    previous = None
    for length in (1, 5, 10, 20, 50):
        infl = rnn_cell.first_token_influence(cell, x0_a, x0_b, rest[: length - 1])
        assert infl <= 0.9 ** (length - 1) * base + 1e-12      # tanh' <= 1 and ||Wh|| = 0.9
        previous = infl
    assert previous < 0.01 * base                              # 0.9^49 is about 0.006
