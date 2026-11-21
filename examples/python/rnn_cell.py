"""A minimal recurrent cell (handout 01).

h_t = tanh(Wx x_t + Wh h_(t-1) + b)

The SAME cell (same weights) is applied at every step. The hidden state h is a fixed-size
vector, so memory per step is constant, but everything the cell knows about the past must
fit into it, and the influence of old tokens fades.
"""
import numpy as np


class RNNCell:
    def __init__(self, d_in, d_h, rng, recurrent_scale=0.9):
        self.Wx = rng.normal(0, 1 / np.sqrt(d_in), (d_h, d_in))
        q, _ = np.linalg.qr(rng.normal(size=(d_h, d_h)))   # orthogonal matrix: keeps vector length
        self.Wh = recurrent_scale * q                      # ...then shrinks it: old info fades by 0.9 per step
        self.b = np.zeros(d_h)

    def step(self, x, h):
        return np.tanh(self.Wx @ x + self.Wh @ h + self.b)

    def run(self, xs):
        """Process a sequence one token at a time. Returns the list of hidden states h_1..h_T."""
        h = np.zeros(self.Wh.shape[0])                     # h_0 = zeros
        states = []
        for x in xs:
            h = self.step(x, h)                            # the loop: h_t depends on h_(t-1)
            states.append(h)
        return states


def first_token_influence(cell, x0_a, x0_b, rest):
    """How much does changing ONLY the first token change the final hidden state?"""
    h_a = cell.run([x0_a, *rest])[-1]
    h_b = cell.run([x0_b, *rest])[-1]
    return float(np.linalg.norm(h_a - h_b))


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    cell = RNNCell(d_in=8, d_h=16, rng=rng)
    tokens = rng.normal(size=(60, 8))

    for n in (3, 300):
        print(f"state size after {n} tokens: {cell.run(rng.normal(size=(n, 8)))[-1].size} numbers (constant)")

    swapped = tokens.copy()
    swapped[[-2, -1]] = swapped[[-1, -2]]          # swap the LAST two tokens (early ones would have faded)
    print("order matters (swap last two tokens): final states differ by",
          round(float(np.linalg.norm(cell.run(tokens)[-1] - cell.run(swapped)[-1])), 3))

    # Small inputs keep tanh in its almost-linear range, so each step scales the old signal by
    # (nearly) exactly the recurrent scale 0.9. With large inputs tanh saturates and the fade is even faster.
    small = 0.02 * tokens
    x0_a, x0_b = 0.02 * rng.normal(size=8), 0.02 * rng.normal(size=8)
    print("\nInfluence of the FIRST token on the final state (recurrent scale 0.9, small inputs):")
    base = first_token_influence(cell, x0_a, x0_b, [])
    for length in (1, 5, 10, 20, 50):
        infl = first_token_influence(cell, x0_a, x0_b, small[: length - 1])
        print(f"  after {length:>2} tokens: {infl:.5f}   (0.9^{length - 1} x first-step change = {0.9 ** (length - 1) * base:.5f})")
