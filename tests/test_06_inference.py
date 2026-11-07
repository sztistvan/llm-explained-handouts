import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples" / "python"))

import batching_sim
import kv_cache
import quantize
import speculative


def test_kv_cache_gives_identical_output_with_less_work():
    rng = np.random.default_rng(0)
    head = kv_cache.Head(16, 8, rng)
    xs = rng.normal(size=(30, 16))
    full = [head.attend_no_cache(xs[: t + 1]) for t in range(30)]
    work_full = head.k_projections
    head.k_projections = 0
    cache = {"K": [], "V": []}
    cached = [head.attend_cached(xs[t], cache) for t in range(30)]
    assert np.allclose(full, cached)
    assert work_full == 30 * 31 // 2 and head.k_projections == 30


def test_kv_cache_size_formula():
    # Llama-3-8B-like: 32 layers, 8 KV heads, head_dim 128, 16-bit -> 128 KiB per token
    assert kv_cache.kv_cache_bytes(32, 8, 128, 1) == 128 * 1024
    assert kv_cache.kv_cache_bytes(32, 8, 128, 8192) == 2**30  # 1 GiB at 8k context


def test_quantization_error_shrinks_with_smaller_groups_and_more_bits():
    rng = np.random.default_rng(0)
    w = rng.normal(0, 0.02, (256, 256))
    w[0, 0] = 0.5

    def err(bits, g):
        q, s = quantize.quantize(w, bits, g)
        return np.abs(quantize.dequantize(q, s, w.shape) - w).mean()

    assert err(8, None) < err(4, None)
    assert err(4, 32) < err(4, None)
    assert err(8, 64) < 0.001


def test_quantization_memory_less_than_float16():
    n = 1_000_000
    assert quantize.memory_bytes(n, 4, 64) < 0.3 * n * 2


def test_speculative_output_matches_target_distribution():
    rng = np.random.default_rng(1)
    p = np.array([0.55, 0.25, 0.15, 0.05])
    q = np.array([0.40, 0.30, 0.20, 0.10])
    n = 100_000
    toks = [speculative.speculative_step(p, q, rng)[0] for _ in range(n)]
    freq = np.bincount(toks, minlength=4) / n
    assert np.allclose(freq, p, atol=0.01)


def test_speculative_acceptance_matches_theory():
    rng = np.random.default_rng(2)
    p = np.array([0.55, 0.25, 0.15, 0.05])
    q = np.array([0.40, 0.30, 0.20, 0.10])
    acc = np.mean([speculative.speculative_step(p, q, rng)[1] for _ in range(50_000)])
    assert acc == pytest.approx(speculative.acceptance_rate(p, q), abs=0.01)
    assert speculative.expected_tokens_per_pass(0.0, 4) == pytest.approx(1.0)
    assert speculative.expected_tokens_per_pass(1.0, 4) == 5


def test_continuous_batching_never_slower_than_static():
    lengths = [5, 400, 7, 9, 300, 4, 6, 8, 10, 12, 450, 3]
    assert batching_sim.continuous_batching(lengths, 4) <= batching_sim.static_batching(lengths, 4)
    assert batching_sim.continuous_batching(lengths, 4) >= sum(lengths) / 4  # work lower bound
