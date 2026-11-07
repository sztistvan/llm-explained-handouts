"""A minimal retrieval-augmented generation (RAG) pipeline (handout 08).

Offline and dependency-light on purpose. The embedding here is a hashed bag-of-words: it
matches WORDS, not MEANING. In production replace `embed` with a real embedding model
(handout 02) and `Index` with a vector database. The pipeline shape stays identical.
"""
import hashlib
import re

import numpy as np

DIM = 512


def chunk_text(text, size=40, overlap=10):
    """Split into word windows of `size` words, each overlapping the previous by `overlap`."""
    words = text.split()
    step = size - overlap
    return [" ".join(words[i:i + size]) for i in range(0, max(len(words) - overlap, 1), step)]


def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def embed(text, dim=DIM):
    v = np.zeros(dim)
    for w in tokenize(text):
        h = int(hashlib.md5(w.encode()).hexdigest(), 16)  # stable across runs (unlike hash())
        v[h % dim] += 1.0
    n = np.linalg.norm(v)
    return v / n if n else v


class Index:
    def __init__(self):
        self.chunks, self.vectors, self.sources = [], [], []

    def add(self, text, source):
        for c in chunk_text(text):
            self.chunks.append(c)
            self.sources.append(source)
            self.vectors.append(embed(c))

    def search(self, query, k=3):
        sims = np.array(self.vectors) @ embed(query)  # cosine, since vectors are unit length
        top = np.argsort(sims)[::-1][:k]
        return [(self.chunks[i], self.sources[i], float(sims[i])) for i in top]


def build_prompt(question, hits):
    context = "\n\n".join(f"[{i + 1}] (source: {src})\n{text}" for i, (text, src, _) in enumerate(hits))
    return (
        "Answer the question using ONLY the sources below. Cite sources like [1]. "
        "If the sources do not contain the answer, say you don't know.\n\n"
        f"<sources>\n{context}\n</sources>\n\n"
        f"Question: {question}"
    )


DOCS = {
    "refunds.md": "Customers can request a refund within 30 days of purchase. Refunds are paid back to the "
                  "original payment method within 5 business days. Digital licenses that were already "
                  "activated cannot be refunded.",
    "shipping.md": "Standard shipping takes 3 to 5 business days within the EU. Express shipping takes 1 to 2 "
                   "business days and costs extra. Orders over 50 euros ship free with standard shipping.",
    "security.md": "Enable two-factor authentication in the account settings. Passwords must have at least 12 "
                   "characters. Sessions expire after 30 minutes of inactivity.",
}

if __name__ == "__main__":
    index = Index()
    for src, text in DOCS.items():
        index.add(text, src)
    question = "How many days do I have to request a refund?"
    hits = index.search(question, k=2)
    for text, src, score in hits:
        print(f"{score:.2f}  {src}: {text[:60]}...")
    print("\n--- prompt sent to the LLM ---")
    print(build_prompt(question, hits))
