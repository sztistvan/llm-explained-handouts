"""Static vs continuous batching, as a step-count simulation (handout 06).

Each request needs `length` decode steps. One step advances every active request by one token.
"""
import random


def static_batching(lengths, batch_size):
    """A batch runs until its LONGEST request finishes; short ones sit idle."""
    steps = 0
    for i in range(0, len(lengths), batch_size):
        steps += max(lengths[i:i + batch_size])
    return steps


def continuous_batching(lengths, batch_size):
    """As soon as a request finishes, its slot is given to the next waiting request."""
    queue = list(lengths)
    active = []
    steps = 0
    while queue or active:
        while queue and len(active) < batch_size:
            active.append(queue.pop(0))
        active = [r - 1 for r in active if r - 1 > 0]
        steps += 1
    return steps


if __name__ == "__main__":
    random.seed(0)
    lengths = [random.choice([10, 20, 50, 400]) for _ in range(64)]  # a few long outliers
    total = sum(lengths)
    for name, fn in (("static", static_batching), ("continuous", continuous_batching)):
        steps = fn(lengths, 8)
        print(f"{name:<11}: {steps:>5} steps, utilization {total / (steps * 8):.0%}")
