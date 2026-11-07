"""Training a tiny language model by gradient descent (handout 05).

The "network" is just a table W[V, V] of logits: row = previous token, column = next token.
The training loop (forward pass, cross-entropy loss, gradient, update) is the same one used
for LLMs. Only the model between input and logits is far bigger.
"""
import numpy as np


def softmax_rows(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def encode(words):
    vocab = sorted(set(words))
    index = {w: i for i, w in enumerate(vocab)}
    return vocab, np.array([index[w] for w in words])


def train(ids, vocab_size, steps=300, lr=1.0):
    X, Y = ids[:-1], ids[1:]           # input token -> target = the NEXT token
    W = np.zeros((vocab_size, vocab_size))
    n = len(X)
    losses = []
    for _ in range(steps):
        P = softmax_rows(W[X])                       # forward: predicted distributions
        loss = -np.log(P[np.arange(n), Y]).mean()    # cross-entropy: -log p(correct token)
        losses.append(loss)
        G = P.copy()
        G[np.arange(n), Y] -= 1                      # d loss / d logits = p - onehot(target)
        G /= n
        gradW = np.zeros_like(W)
        np.add.at(gradW, X, G)                       # accumulate gradient per table row
        W -= lr * gradW                              # gradient descent step
    return W, losses


if __name__ == "__main__":
    words = "the cat sat on the mat the cat ate the fish".split()
    vocab, ids = encode(words)
    W, losses = train(ids, len(vocab))
    print(f"loss: start {losses[0]:.3f} -> end {losses[-1]:.3f}  "
          f"(perplexity {np.exp(losses[0]):.1f} -> {np.exp(losses[-1]):.2f})")
    the = vocab.index("the")
    probs = softmax_rows(W[[the]])[0]
    print("after 'the':", {w: round(float(p), 2) for w, p in zip(vocab, probs) if p > 0.01})
