"""Quality gates for the Markdown handouts: internal links resolve, code fences are balanced,
every Mermaid block starts with a known diagram type, and every handout is indexed."""
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MD_FILES = sorted([ROOT / "README.md", ROOT / "SCOPE.md", ROOT / "glossary.md", *(ROOT / "docs").glob("*.md")])
HANDOUTS = sorted((ROOT / "docs").glob("[0-9][0-9]-*.md"))
MERMAID_TYPES = ("flowchart", "sequenceDiagram", "timeline", "graph", "classDiagram", "stateDiagram", "gantt", "pie")


def strip_code(text):
    return re.sub(r"```.*?```", "", text, flags=re.S)


@pytest.mark.parametrize("md", MD_FILES, ids=lambda p: p.name)
def test_relative_links_resolve(md):
    for target in re.findall(r"\]\(([^)\s]+)\)", strip_code(md.read_text(encoding="utf-8"))):
        if target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        path = (md.parent / target.split("#")[0]).resolve()
        assert path.exists(), f"{md.name}: broken link -> {target}"


@pytest.mark.parametrize("md", MD_FILES, ids=lambda p: p.name)
def test_code_fences_are_balanced(md):
    fences = [l for l in md.read_text(encoding="utf-8").splitlines() if l.startswith("```")]
    assert len(fences) % 2 == 0, f"{md.name}: unbalanced ``` fences"


@pytest.mark.parametrize("md", HANDOUTS, ids=lambda p: p.name)
def test_mermaid_blocks_start_with_a_known_type(md):
    for block in re.findall(r"```mermaid\n(.*?)```", md.read_text(encoding="utf-8"), flags=re.S):
        assert block.lstrip().startswith(MERMAID_TYPES), f"{md.name}: unknown mermaid type: {block[:30]!r}"


def test_every_handout_is_in_the_readme_and_glossary_header():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    glossary = (ROOT / "glossary.md").read_text(encoding="utf-8")
    for h in HANDOUTS:
        if h.name != "10-further-reading.md":
            assert f"docs/{h.name}" in readme, f"{h.name} missing from README"
            assert f"docs/{h.name}" in glossary, f"{h.name} missing from glossary header"
        else:
            assert f"docs/{h.name}" in readme


@pytest.mark.parametrize("md", HANDOUTS[:10], ids=lambda p: p.name)
def test_every_further_reading_entry_has_a_web_link(md):
    text = md.read_text(encoding="utf-8")
    section = text.split("## Further reading", 1)[1]
    entries = [l for l in section.splitlines() if l.startswith("- ")]
    assert entries, f"{md.name}: empty Further reading"
    for line in entries:
        assert re.search(r"\]\(https?://", line), f"{md.name}: no link in: {line[:70]}"


sys.path.insert(0, str(ROOT / "tools"))
import extract_math  # noqa: E402


@pytest.mark.parametrize("md", extract_math.FILES, ids=lambda p: p.name)
def test_math_delimiters_are_balanced_and_currency_is_escaped(md):
    _, _, unmatched = extract_math.extract(md.read_text(encoding="utf-8"))
    assert unmatched == 0, f"{md.name}: {unmatched} unmatched $ (escape currency as \\$)"


@pytest.mark.parametrize("md", extract_math.FILES, ids=lambda p: p.name)
def test_no_raw_pipe_inside_math_in_tables(md):
    """A '|' inside $...$ on a table row would split the cell. Use \\mid or \\lVert instead."""
    for line in md.read_text(encoding="utf-8").splitlines():
        if line.lstrip().startswith("|"):
            for formula in re.findall(r"(?<!\\)\$([^$]+?)\$", line):
                assert "|" not in formula, f"{md.name}: raw | in math within table row: {line[:70]}"


FORBIDDEN_MACROS = [r"\operatorname"]  # rejected by GitHub's renderer ("macros are not allowed")


@pytest.mark.parametrize("md", extract_math.FILES, ids=lambda p: p.name)
def test_no_macros_that_github_rejects(md):
    text = md.read_text(encoding="utf-8")
    for macro in FORBIDDEN_MACROS:
        assert macro not in text, f"{md.name}: {macro} is not allowed on GitHub (use \\mathrm or \\text)"


@pytest.mark.parametrize("md", extract_math.FILES, ids=lambda p: p.name)
def test_display_math_has_blank_lines_around_it(md):
    """GitHub only treats $$ ... $$ as a display block with a blank line before and after it."""
    lines = md.read_text(encoding="utf-8").splitlines()
    in_code, open_at = False, None
    for i, line in enumerate(lines):
        if line.startswith("```"):
            in_code = not in_code
        if in_code or line.strip() != "$$":
            continue
        if open_at is None:                                   # opening fence
            open_at = i
            assert i == 0 or lines[i - 1].strip() == "", f"{md.name}:{i + 1}: blank line needed before $$"
        else:                                                 # closing fence
            nxt = lines[i + 1] if i + 1 < len(lines) else ""
            assert nxt.strip() == "", f"{md.name}:{i + 1}: blank line needed after closing $$"
            open_at = None
    assert open_at is None, f"{md.name}: unclosed $$ block"


INLINE_MATH = re.compile(r"(?<![\\$])\$(?!\$)([^$\n]+?)(?<![\\$])\$(?!\$)")


@pytest.mark.parametrize("md", extract_math.FILES, ids=lambda p: p.name)
def test_inline_math_follows_githubs_delimiter_rules(md):
    text = extract_math.strip_code(md.read_text(encoding="utf-8"))
    text = re.sub(r"\$\$.+?\$\$", "", text, flags=re.S)       # ignore display blocks
    for m in INLINE_MATH.finditer(text):
        before = text[m.start() - 1] if m.start() > 0 else " "
        after = text[m.end()] if m.end() < len(text) else " "
        where = f"{md.name}: ...{text[max(0, m.start() - 25):m.end() + 6]!r}"
        assert not (before.isalnum() or before == "-"), f"{where}: opening $ glued to text, add a space"
        assert not after.isalnum(), f"{where}: closing $ glued to a letter or digit"
        assert not (m.group(1).endswith(")") and after == ")"), f"{where}: ')$)' is not parsed as math by GitHub"


def test_formulas_are_present_in_the_math_heavy_handouts():
    counts = {name: len(display) + len(inline) for name, display, inline, _ in extract_math.all_math()}
    assert counts["03-transformer.md"] > 50
    assert counts["04-generation.md"] > 10
    assert counts["glossary.md"] > 20


def test_every_example_file_referenced_in_docs_exists():
    for md in HANDOUTS:
        for ref in re.findall(r"\.\./(examples/[\w/.\-]+)", md.read_text(encoding="utf-8")):
            assert (ROOT / ref).exists(), f"{md.name}: missing example {ref}"
