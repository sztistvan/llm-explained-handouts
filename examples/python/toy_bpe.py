"""Toy Byte-Pair Encoding (BPE) tokenizer, character-level for readability.

Real tokenizers (GPT, LLaMA, ...) work on bytes and use tuned rules, but the
core algorithm is the same: repeatedly merge the most frequent adjacent pair.
"""
from collections import Counter


def count_pairs(tokens):
    return Counter(zip(tokens, tokens[1:]))


def merge_pair(tokens, pair, new_token):
    out, i = [], 0
    while i < len(tokens):
        if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
            out.append(new_token)
            i += 2
        else:
            out.append(tokens[i])
            i += 1
    return out


def train_bpe(text, num_merges):
    tokens = list(text)
    merges = []  # ordered list of (pair, new_token): this IS the tokenizer
    for _ in range(num_merges):
        pairs = count_pairs(tokens)
        if not pairs:
            break
        best = pairs.most_common(1)[0][0]
        new_token = best[0] + best[1]
        tokens = merge_pair(tokens, best, new_token)
        merges.append((best, new_token))
    return merges


def encode(text, merges):
    tokens = list(text)
    for pair, new_token in merges:  # apply merges in training order
        tokens = merge_pair(tokens, pair, new_token)
    return tokens


if __name__ == "__main__":
    corpus = "low lower lowest newer newest wider widest low low"
    merges = train_bpe(corpus, num_merges=10)
    print("Learned merges:")
    for (a, b), new in merges:
        print(f"  {a!r} + {b!r} -> {new!r}")
    print()
    for word in ["lowest", "newest", "slowest"]:  # 'slowest' was never seen
        print(f"{word!r:10} -> {encode(word, merges)}")
