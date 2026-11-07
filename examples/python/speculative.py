"""Speculative decoding, one step (handout 06).

A cheap draft model proposes a token x ~ q. The big target model checks it:
accept with probability min(1, p(x)/q(x)); otherwise resample from the "residual"
distribution max(0, p - q) normalized. The result is distributed EXACTLY like p, so quality
is unchanged, and when the draft is good most proposals are accepted, so the big model
verifies several tokens per (expensive) pass.
"""
import numpy as np


def speculative_step(p, q, rng):
    """Return (token, accepted_draft). p = target probs, q = draft probs (same vocab)."""
    x = rng.choice(len(q), p=q)
    if rng.random() < min(1.0, p[x] / q[x]):
        return int(x), True
    residual = np.maximum(p - q, 0)
    residual /= residual.sum()
    return int(rng.choice(len(p), p=residual)), False


def acceptance_rate(p, q):
    """Probability that a draft token is accepted = sum_x min(p(x), q(x))."""
    return float(np.minimum(p, q).sum())


def expected_tokens_per_pass(alpha, k):
    """Expected tokens produced per target pass with k drafted tokens and acceptance rate alpha."""
    if alpha == 1.0:
        return k + 1
    return (1 - alpha ** (k + 1)) / (1 - alpha)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    p = np.array([0.55, 0.25, 0.15, 0.05])  # target
    q = np.array([0.40, 0.30, 0.20, 0.10])  # draft (similar but not equal)
    n = 200_000
    draws = [speculative_step(p, q, rng) for _ in range(n)]
    freq = np.bincount([t for t, _ in draws], minlength=4) / n
    print("target p        :", p)
    print("empirical output:", np.round(freq, 3), " <- matches p, not q")
    print("acceptance rate :", round(np.mean([a for _, a in draws]), 3),
          "(theory", round(acceptance_rate(p, q), 3), ")")
    for k in (2, 4):
        print(f"k={k}: ~{expected_tokens_per_pass(acceptance_rate(p, q), k):.2f} tokens per target pass")
