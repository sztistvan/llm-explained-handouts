"""Cost estimation and retry with exponential backoff (handout 09).

Prices below are ILLUSTRATIVE. Use your provider's current price list.
"""
import random
import time


def estimate_cost(input_tokens, output_tokens, price_in_per_m, price_out_per_m,
                  cached_tokens=0, cached_discount=0.9):
    """Cost in currency units. Cached input tokens are billed at (1 - cached_discount) of the price."""
    fresh = input_tokens - cached_tokens
    input_cost = (fresh + cached_tokens * (1 - cached_discount)) * price_in_per_m / 1e6
    return input_cost + output_tokens * price_out_per_m / 1e6


def retry_with_backoff(fn, retries=5, base=0.5, factor=2.0, jitter=0.1,
                       sleep=time.sleep, rng=random.random, retry_on=(Exception,)):
    """Call fn(); on failure wait base * factor**attempt (+ random jitter) and retry.
    `sleep` and `rng` are injectable so tests do not really wait."""
    for attempt in range(retries + 1):
        try:
            return fn()
        except retry_on:
            if attempt == retries:
                raise
            sleep(base * factor ** attempt * (1 + jitter * rng()))


if __name__ == "__main__":
    plain = estimate_cost(10_000, 500, 3.0, 15.0)
    cached = estimate_cost(10_000, 500, 3.0, 15.0, cached_tokens=8_000)
    print(f"request: 10k input + 500 output tokens -> {plain:.4f} (no cache), {cached:.4f} (8k tokens cached)")
    print(f"per 100,000 requests: {plain * 100_000:,.0f} vs {cached * 100_000:,.0f}")

    delays, attempts = [], []

    def flaky():
        attempts.append(1)
        if len(attempts) < 4:
            raise TimeoutError("rate limited")
        return "ok"

    print(retry_with_backoff(flaky, sleep=delays.append, rng=lambda: 0.0), "after waits", delays)
