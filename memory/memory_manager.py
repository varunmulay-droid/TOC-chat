"""
memory/memory_manager.py
==========================
Conversation memory: after each response, embed the (user, assistant)
exchange and store it. On later turns, retrieve the most relevant past
exchanges as extra context.
"""

from typing import List, Optional

from memory.faiss_store import FaissStore
from models.encoder import MiniLMEncoder


class MemoryManager:
    def __init__(self, encoder: Optional[MiniLMEncoder] = None):
        self.encoder = encoder or MiniLMEncoder()
        self.store = FaissStore()

    def remember(self, user_text: str, assistant_text: str):
        combined = f"User: {user_text}\nAssistant: {assistant_text}"
        vector = self.encoder.encode(combined)
        self.store.add(vector, texts=[combined], metadatas=[{"type": "exchange"}])

    def recall(self, query: str, k: int = 3) -> List[str]:
        vector = self.encoder.encode(query)
        results = self.store.search(vector, k=k)
        return [item.text for item, _score in results]

    def save(self, path: str = "data/conversations/memory"):
        self.store.save(path)

    def load(self, path: str = "data/conversations/memory"):
        self.store.load(path)
