"""Decoding strategies: greedy, temperature, top-k, top-p, beam search (handout 04).

`logits` is a 1-D array with one score per vocabulary token.
"""
import math

import numpy as np


def softmax(x):
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def apply_temperature(logits, temperature):
    return logits / temperature  # <1 sharpens, >1 flattens


def top_k_filter(probs, k):
    """Keep only the k most probable tokens, renormalize."""
    out = np.zeros_like(probs)
    keep = np.argsort(probs)[::-1][:k]
    out[keep] = probs[keep]
    return out / out.sum()


def top_p_filter(probs, p):
    """Nucleus sampling: keep the smallest set of top tokens whose total probability >= p."""
    order = np.argsort(probs)[::-1]
    cumulative = np.cumsum(probs[order])
    cutoff = np.searchsorted(cumulative, p) + 1  # first index where cumulative >= p, inclusive
    keep = order[:cutoff]
    out = np.zeros_like(probs)
    out[keep] = probs[keep]
    return out / out.sum()


def next_token_distribution(logits, temperature=1.0, top_k=None, top_p=None):
    """Order used by most libraries: temperature -> top-k -> top-p."""
    probs = softmax(apply_temperature(logits, temperature))
    if top_k is not None:
        probs = top_k_filter(probs, top_k)
    if top_p is not None:
        probs = top_p_filter(probs, top_p)
    return probs


def sample(logits, temperature=1.0, top_k=None, top_p=None, rng=None):
    rng = rng or np.random.default_rng()
    if temperature == 0:  # convention: temperature 0 means greedy
        return int(np.argmax(logits))
    probs = next_token_distribution(logits, temperature, top_k, top_p)
    return int(rng.choice(len(probs), p=probs))


# ---- Greedy vs beam search on a toy model -------------------------------------------------

def greedy_search(step_fn, steps):
    seq, logp = [], 0.0
    for _ in range(steps):
        dist = step_fn(tuple(seq))
        tok = max(dist, key=dist.get)
        seq.append(tok)
        logp += math.log(dist[tok])
    return seq, logp


def beam_search(step_fn, steps, width):
    """step_fn(seq) -> {token: probability}. Keeps the `width` best sequences by total log-prob."""
    beams = [([], 0.0)]
    for _ in range(steps):
        candidates = []
        for seq, logp in beams:
            for tok, p in step_fn(tuple(seq)).items():
                candidates.append((seq + [tok], logp + math.log(p)))
        beams = sorted(candidates, key=lambda b: b[1], reverse=True)[:width]
    return beams


def toy_model(seq):
    """Greedy takes 'A' first (0.6) but the best full sequence starts with 'B'."""
    if seq == ():
        return {"A": 0.6, "B": 0.4}
    if seq == ("A",):
        return {"x": 0.5, "y": 0.5}
    if seq == ("B",):
        return {"z": 0.9, "w": 0.1}
    raise ValueError(seq)


if __name__ == "__main__":
    vocab = ["mat", "sofa", "moon", "banana"]
    logits = np.array([3.0, 2.0, 0.5, -1.0])

    print("Token probabilities at different temperatures:")
    for t in (0.2, 1.0, 2.0):
        probs = next_token_distribution(logits, temperature=t)
        print(f"  T={t}:", {v: round(float(p), 3) for v, p in zip(vocab, probs)})

    print("\nTop-k=2 :", np.round(next_token_distribution(logits, top_k=2), 3))
    print("Top-p=0.9:", np.round(next_token_distribution(logits, top_p=0.9), 3))

    rng = np.random.default_rng(0)
    picks = [vocab[sample(logits, 1.0, rng=rng)] for _ in range(10)]
    print("\n10 samples at T=1:", picks)

    print("\nGreedy:", greedy_search(toy_model, 2))
    print("Beam(2):", beam_search(toy_model, 2, 2)[0])
