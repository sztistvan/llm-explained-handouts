"""A tiny evaluation harness for LLM features (handout 09).

Ideas shown: programmatic checks, a fixed test set, repeated trials (outputs are random),
comparing two versions, and a regression gate for CI. The "model" here is a mock with a
configurable accuracy so the example runs offline and deterministically.
"""
import json
import random
import statistics


# ---- checks: take the model output string, return True/False -------------------------------
def exact(expected):
    return lambda out: out.strip() == expected


def contains(text):
    return lambda out: text.lower() in out.lower()


def json_with_keys(*keys):
    def check(out):
        try:
            data = json.loads(out)
        except ValueError:
            return False
        return isinstance(data, dict) and all(k in data for k in keys)
    return check


# ---- harness ---------------------------------------------------------------------------------
def run_suite(model_fn, cases):
    """cases: list of (prompt, check). Returns the fraction passed in ONE run of the suite."""
    return sum(check(model_fn(prompt)) for prompt, check in cases) / len(cases)


def run_repeated(model_factory, cases, runs):
    """Outputs vary between runs, so measure several. model_factory(seed) builds a fresh model."""
    return [run_suite(model_factory(seed), cases) for seed in range(runs)]


def summarize(scores):
    return {"mean": statistics.mean(scores), "min": min(scores), "max": max(scores),
            "stdev": statistics.pstdev(scores)}


def regression_gate(candidate_scores, baseline_mean, tolerance=0.02):
    """Fail the build if the candidate's mean score drops below the baseline minus a tolerance."""
    return statistics.mean(candidate_scores) >= baseline_mean - tolerance


# ---- mock model for the demo and tests ------------------------------------------------------
ANSWERS = {f"What is {a}+{b}?": str(a + b) for a, b in [(2, 2), (3, 5), (7, 6), (9, 9), (10, 4),
                                                       (12, 3), (8, 8), (5, 1), (6, 6), (4, 9)]}
CASES = [(prompt, exact(answer)) for prompt, answer in ANSWERS.items()]


def make_mock_model(accuracy):
    def factory(seed):
        rng = random.Random(seed)

        def model(prompt):
            return ANSWERS[prompt] if rng.random() < accuracy else "I am not sure"
        return model
    return factory


if __name__ == "__main__":
    v1, v2 = make_mock_model(0.70), make_mock_model(0.85)
    print("single run each (noisy):   v1 =", run_suite(v1(0), CASES), " v2 =", run_suite(v2(0), CASES))
    s1, s2 = summarize(run_repeated(v1, CASES, 30)), summarize(run_repeated(v2, CASES, 30))
    for name, s in (("v1", s1), ("v2", s2)):
        print(f"30 runs, {name}: mean {s['mean']:.2f}  range {s['min']:.1f}-{s['max']:.1f}  stdev {s['stdev']:.2f}")
    print("gate (candidate v1 vs baseline v2):", regression_gate(run_repeated(v1, CASES, 30), s2["mean"]))
