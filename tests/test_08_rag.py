import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples" / "python"))

import rag


def test_chunking_overlaps_and_covers_text():
    text = " ".join(f"w{i}" for i in range(100))
    chunks = rag.chunk_text(text, size=30, overlap=10)
    assert all(len(c.split()) <= 30 for c in chunks)
    assert chunks[0].split()[-10:] == chunks[1].split()[:10]          # overlap preserved
    assert {w for c in chunks for w in c.split()} == set(text.split())  # nothing lost


def test_embedding_is_deterministic_and_unit_length():
    a, b = rag.embed("refund within 30 days"), rag.embed("refund within 30 days")
    assert np.array_equal(a, b)
    assert np.linalg.norm(a) == pytest.approx(1.0)


def test_retrieval_finds_the_relevant_document():
    index = rag.Index()
    for src, text in rag.DOCS.items():
        index.add(text, src)
    hits = index.search("How many days to get a refund for my purchase?", k=1)
    assert hits[0][1] == "refunds.md"
    hits = index.search("how long does express shipping take", k=1)
    assert hits[0][1] == "shipping.md"


def test_prompt_contains_sources_question_and_grounding_rule():
    index = rag.Index()
    index.add(rag.DOCS["refunds.md"], "refunds.md")
    hits = index.search("refund", k=1)
    prompt = rag.build_prompt("Can I get a refund?", hits)
    assert "refunds.md" in prompt and "Can I get a refund?" in prompt
    assert "don't know" in prompt and "<sources>" in prompt


def test_lexical_embedding_limitation_is_real():
    """Documented limitation: no shared words -> no similarity, even if the meaning matches."""
    sim = rag.embed("money back") @ rag.embed("refund")
    assert sim == pytest.approx(0.0)
