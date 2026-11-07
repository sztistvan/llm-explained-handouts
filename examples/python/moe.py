"""Mixture of Experts layer with top-k routing (handout 07).

Replaces the dense MLP of a Transformer block. A small router picks k of n experts per
token; only those experts run. Total parameters grow with n, compute per token with k.
"""
import numpy as np


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


class MoE:
    def __init__(self, d_model, d_hidden, n_experts, k, rng):
        self.n, self.k = n_experts, k
        self.router = rng.normal(0, 0.5, (d_model, n_experts))
        self.experts = [
            (rng.normal(0, 0.2, (d_model, d_hidden)), rng.normal(0, 0.2, (d_hidden, d_model)))
            for _ in range(n_experts)
        ]

    @staticmethod
    def _expert(x, W1, W2):
        return np.maximum(x @ W1, 0) @ W2  # tiny ReLU MLP

    def forward(self, x):
        """x: (tokens, d_model). Returns (output, tokens_per_expert)."""
        logits = x @ self.router                                   # (tokens, n_experts)
        top = np.argsort(logits, axis=1)[:, -self.k:]              # indices of the k best experts
        out = np.zeros_like(x)
        load = np.zeros(self.n, dtype=int)
        for t in range(len(x)):
            gates = softmax(logits[t, top[t]])                     # weights over the chosen k only
            for g, e in zip(gates, top[t]):
                out[t] += g * self._expert(x[t], *self.experts[e])  # only k experts are computed
                load[e] += 1
        return out, load

    def param_counts(self):
        d, h = self.experts[0][0].shape
        per_expert = 2 * d * h
        router = d * self.n
        return {"total": self.n * per_expert + router, "active_per_token": self.k * per_expert + router}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    moe = MoE(d_model=16, d_hidden=64, n_experts=8, k=2, rng=rng)
    x = rng.normal(size=(1000, 16))
    _, load = moe.forward(x)
    print("tokens routed to each expert:", load.tolist(), "(sum =", load.sum(), "= 1000 tokens x k=2)")
    c = moe.param_counts()
    print(f"parameters: total {c['total']:,}; used per token {c['active_per_token']:,} "
          f"({c['active_per_token'] / c['total']:.0%})")
