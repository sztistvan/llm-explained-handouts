"""Toy embeddings and cosine similarity (handout 02)."""
import numpy as np

emb = {
    "king": np.array([0.9, 0.8, 0.1]),
    "queen": np.array([0.9, 0.1, 0.8]),
    "apple": np.array([0.1, -0.5, 0.2]),
}


def cosine(a, b):
    return a @ b / (np.linalg.norm(a) * np.linalg.norm(b))


if __name__ == "__main__":
    print(cosine(emb["king"], emb["queen"]))
    print(cosine(emb["king"], emb["apple"]))
