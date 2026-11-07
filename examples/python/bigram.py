"""Toy bigram language model (handout 00)."""
import random
from collections import defaultdict, Counter


def train(words):
    counts = defaultdict(Counter)
    for a, b in zip(words, words[1:]):
        counts[a][b] += 1
    return counts


def next_word(counts, prev, rng=random):
    words, weights = zip(*counts[prev].items())
    return rng.choices(words, weights=weights)[0]


def generate(counts, start, n, rng=random):
    out = [start]
    for _ in range(n):
        if not counts[out[-1]]:  # dead end: word never had a successor
            break
        out.append(next_word(counts, out[-1], rng))
    return out


if __name__ == "__main__":
    text = "the cat sat on the mat the cat ate the fish".split()
    print(" ".join(generate(train(text), "the", 6)))
