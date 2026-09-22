"""
memory/faiss_store.py
======================
A single shared FAISS index wrapper, used for BOTH document RAG chunks
and conversation memory (kept as two separate named indices via
FaissStore instances, but identical implementation -- no need for two
different vector store classes).

Lives on CPU per the T4 memory strategy.
"""

import os
import pickle
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import faiss

from models.encoder import EMBEDDING_DIM


@dataclass
class StoredItem:
    text: str
    metadata: dict = field(default_factory=dict)


class FaissStore:
    def __init__(self, dim: int = EMBEDDING_DIM):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # inner product == cosine sim, since embeddings are normalized
        self.items: List[StoredItem] = []

    def add(self, vectors: np.ndarray, texts: List[str], metadatas: List[dict] | None = None):
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        metadatas = metadatas or [{} for _ in texts]
        self.index.add(vectors.astype(np.float32))
        for text, meta in zip(texts, metadatas):
            self.items.append(StoredItem(text=text, metadata=meta))

    def search(self, query_vector: np.ndarray, k: int = 4) -> List[Tuple[StoredItem, float]]:
        if self.index.ntotal == 0:
            return []
        query_vector = query_vector.reshape(1, -1).astype(np.float32)
        k = min(k, self.index.ntotal)
        scores, idxs = self.index.search(query_vector, k)
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            results.append((self.items[idx], float(score)))
        return results

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        faiss.write_index(self.index, path + ".index")
        with open(path + ".items.pkl", "wb") as f:
            pickle.dump(self.items, f)

    def load(self, path: str):
        self.index = faiss.read_index(path + ".index")
        with open(path + ".items.pkl", "rb") as f:
            self.items = pickle.load(f)


if __name__ == "__main__":
    # Smoke test with random vectors (no model load required)
    store = FaissStore(dim=8)
    vecs = np.random.rand(3, 8).astype(np.float32)
    store.add(vecs, texts=["a", "b", "c"])
    results = store.search(vecs[0], k=2)
    print("Search results:", [(item.text, round(score, 3)) for item, score in results])
