"""Quality gates for the Markdown handouts: internal links resolve, code fences are balanced,
every Mermaid block starts with a known diagram type, and every handout is indexed."""
import re
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


def test_every_example_file_referenced_in_docs_exists():
    for md in HANDOUTS:
        for ref in re.findall(r"\.\./(examples/[\w/.\-]+)", md.read_text(encoding="utf-8")):
            assert (ROOT / ref).exists(), f"{md.name}: missing example {ref}"
