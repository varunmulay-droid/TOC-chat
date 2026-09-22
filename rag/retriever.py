"""
rag/retriever.py
=================
End-to-end RAG: ingest a document (load -> chunk -> embed -> store),
and retrieve the most relevant chunks for a query.
"""

from typing import List, Optional

from memory.faiss_store import FaissStore
from models.encoder import MiniLMEncoder
from rag.loader import load_document
from rag.chunker import chunk_text


class Retriever:
    def __init__(self, encoder: Optional[MiniLMEncoder] = None):
        self.encoder = encoder or MiniLMEncoder()
        self.store = FaissStore()

    def ingest(self, path: str, source_name: Optional[str] = None):
        text = load_document(path)
        chunks = chunk_text(text)
        if not chunks:
            return 0
        vectors = self.encoder.encode(chunks)
        metadatas = [{"source": source_name or path} for _ in chunks]
        self.store.add(vectors, texts=chunks, metadatas=metadatas)
        return len(chunks)

    def retrieve(self, query: str, k: int = 4) -> List[str]:
        vector = self.encoder.encode(query)
        results = self.store.search(vector, k=k)
        return [item.text for item, _score in results]

    def save(self, path: str = "data/documents/rag_index"):
        self.store.save(path)

    def load(self, path: str = "data/documents/rag_index"):
        self.store.load(path)
