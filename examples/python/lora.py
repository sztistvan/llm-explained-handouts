"""LoRA (low-rank adaptation) of one linear layer (handout 05).

Instead of updating the big frozen matrix W0 (d_out x d_in), train two small matrices:
B (d_out x r) and A (r x d_in), with rank r << d. The effective weight is
W0 + (alpha / r) * B @ A.
"""
import numpy as np


class LoRALinear:
    def __init__(self, W0, r, alpha, rng):
        self.W0 = W0                                        # frozen pretrained weight
        d_out, d_in = W0.shape
        self.A = rng.normal(0, 0.01, (r, d_in))             # trainable
        self.B = np.zeros((d_out, r))                       # trainable, zero => starts as the base model
        self.scale = alpha / r

    def forward(self, x):
        return x @ self.W0.T + self.scale * (x @ self.A.T) @ self.B.T

    def merged_weight(self):
        """Fold the adapter into one matrix for serving (no extra latency)."""
        return self.W0 + self.scale * self.B @ self.A

    def trainable_params(self):
        return self.A.size + self.B.size


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    d = 4096
    layer = LoRALinear(rng.normal(size=(d, d)) * 0.02, r=8, alpha=16, rng=rng)
    full, lora = layer.W0.size, layer.trainable_params()
    print(f"full matrix: {full:,} params; LoRA r=8: {lora:,} params ({100 * lora / full:.2f}%)")
