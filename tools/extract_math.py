r"""Extract LaTeX math ($...$ inline, $$...$$ display) from the Markdown handouts.

Used by tests (balanced delimiters) and for rendering checks. Code fences and inline code are
ignored. A dollar sign written as \$ is a literal dollar (for example a price).
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = [*sorted((ROOT / "docs").glob("*.md")), ROOT / "glossary.md"]
LITERAL_DOLLAR = "\\$"


def strip_code(text):
    text = re.sub(r"```.*?```", "", text, flags=re.S)  # fenced code
    return re.sub(r"`[^`\n]*`", "", text)  # inline code


def extract(text):
    """Return (display_formulas, inline_formulas, unmatched_dollar_count) for one Markdown text."""
    text = strip_code(text).replace(LITERAL_DOLLAR, "\x00")  # protect literal dollars
    display = [m.strip() for m in re.findall(r"\$\$(.+?)\$\$", text, flags=re.S)]
    rest = re.sub(r"\$\$.+?\$\$", "", text, flags=re.S)
    inline = [m.strip() for m in re.findall(r"\$([^$\n]+?)\$", rest)]
    unmatched = rest.count("$") - 2 * len(inline)
    return display, inline, unmatched


def all_math():
    out = []
    for f in FILES:
        display, inline, unmatched = extract(f.read_text(encoding="utf-8"))
        out.append((f.name, display, inline, unmatched))
    return out


if __name__ == "__main__":
    total_d = total_i = 0
    for name, display, inline, unmatched in all_math():
        print(f"{name:34} display={len(display):3} inline={len(inline):3} unmatched$={unmatched}")
        total_d += len(display)
        total_i += len(inline)
    print("TOTAL display", total_d, "inline", total_i)
