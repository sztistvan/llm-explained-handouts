"""Validate model output and retry with the error message (handout 09).

Never trust model output: parse it, check it against a schema and business rules, and on
failure ask again (a limited number of times) while telling the model what was wrong.
"""
import json

SCHEMA = {
    "category": ("enum", ["billing", "bug", "feature_request", "other"]),
    "reason": str,
}


def validate(data, schema):
    """Return a list of human-readable errors (empty list = valid)."""
    if not isinstance(data, dict):
        return ["output must be a JSON object"]
    errors = []
    for field, rule in schema.items():
        if field not in data:
            errors.append(f"missing field '{field}'")
        elif isinstance(rule, tuple) and rule[0] == "enum":
            if data[field] not in rule[1]:
                errors.append(f"'{field}' must be one of {rule[1]}, got {data[field]!r}")
        elif isinstance(rule, type) and not isinstance(data[field], rule):
            errors.append(f"'{field}' must be {rule.__name__}")
    return errors


def parse_json(text):
    """Models sometimes wrap JSON in code fences. Strip them, then parse. Returns (data, error)."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned[cleaned.find("{"):] if "{" in cleaned else cleaned
    try:
        return json.loads(cleaned), None
    except ValueError as e:
        return None, f"invalid JSON: {e}"


def call_with_retries(model_fn, prompt, schema, max_attempts=3):
    """Returns (data, attempts_used). Raises ValueError after max_attempts failures."""
    last_errors = []
    for attempt in range(1, max_attempts + 1):
        text = model_fn(prompt if not last_errors else
                        prompt + "\n\nYour previous reply was rejected: " + "; ".join(last_errors) +
                        "\nReply again with JSON only.")
        data, err = parse_json(text)
        last_errors = [err] if err else validate(data, schema)
        if not last_errors:
            return data, attempt
    raise ValueError(f"no valid output after {max_attempts} attempts: {last_errors}")


if __name__ == "__main__":
    replies = iter([
        "Sure! The category is billing.",                                   # not JSON
        '{"category": "payment", "reason": "double charge"}',               # JSON, but invalid enum
        '```json\n{"category": "billing", "reason": "double charge"}\n```',  # valid, in a code fence
    ])
    data, attempts = call_with_retries(lambda p: next(replies), "Classify: I was charged twice.", SCHEMA)
    print(f"valid after {attempts} attempts:", data)
