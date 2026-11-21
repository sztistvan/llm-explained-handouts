"""Check that GitHub itself recognizes every formula as math. Needs network.

    python tools/check_github_math.py

All Markdown files are rendered together with GitHub's Markdown API (the same parser used on
github.com), and every formula found by tools/extract_math.py must come back as a
<math-renderer> element. This catches parsing problems that a KaTeX check cannot, for example a
display block without a blank line before and after it, or an inline formula ending in ")$)".

It cannot see problems that GitHub's browser-side renderer reports later (such as disallowed
macros like \\operatorname); tests/test_docs.py forbids the known ones.
The unauthenticated API allows about 60 requests per hour, so this makes a single request.
"""
import html
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import extract_math  # noqa: E402


def render(markdown):
    out = subprocess.run(
        ["curl", "-s", "-m", "90", "-X", "POST", "https://api.github.com/markdown",
         "-H", "Accept: application/vnd.github+json", "-H", "User-Agent: llm-handouts-math-check",
         "-H", "Content-Type: application/json", "--data-binary", "@-"],   # body via stdin: avoids the OS command-line limit
        input=json.dumps({"text": markdown, "mode": "gfm"}),
        capture_output=True, text=True, encoding="utf-8")
    if out.stdout.lstrip().startswith("{"):  # a JSON error, for example rate limiting
        raise RuntimeError(out.stdout[:200])
    return out.stdout


def normalize(tex):
    """Keep letters and digits only: the API output may re-escape characters such as > and \\,."""
    previous = None
    while previous != tex:                       # the API can escape twice (&amp;gt;)
        previous, tex = tex, html.unescape(tex)
    return re.sub(r"[^A-Za-z0-9]", "", tex)


def main():
    expected = []                                 # (file name, formula)
    parts = []
    for md in extract_math.FILES:
        text = md.read_text(encoding="utf-8")
        display, inline, _ = extract_math.extract(text)
        expected += [(md.name, f) for f in display + inline]
        parts.append(text)
    rendered = render("\n\n---\n\n".join(parts))
    found = Counter(normalize(m) for m in re.findall(r"<math-renderer[^>]*>(.*?)</math-renderer>", rendered, flags=re.S))
    if not found:
        raise RuntimeError("GitHub returned no math elements at all (rate limit or API change?)")
    missing = []
    for name, formula in expected:                # multiset match, so repeated one-letter formulas count
        key = normalize(formula)
        if found[key] > 0:
            found[key] -= 1
        else:
            missing.append((name, formula))
    print(f"{len(expected)} formulas in {len(extract_math.FILES)} files; GitHub recognized {len(expected) - len(missing)}")
    for name, formula in missing:
        print(f"  NOT recognized in {name}: {formula[:90]!r}")
    print("all formulas recognized by GitHub" if not missing else f"{len(missing)} problem(s)")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
