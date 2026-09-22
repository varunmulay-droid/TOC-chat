"""
models/encoder.py
==================
all-MiniLM-L6-v2 wrapper for 384-dim embeddings used by both RAG and
conversation memory. Defaults to CPU (per the T4 memory strategy: keep
MiniLM off the GPU unless there's headroom) with an opt-in GPU flag.
"""

import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


@dataclass
class EncoderConfig:
    device: str = os.environ.get("TOC_GPT_ENCODER_DEVICE", "cpu")  # "cpu" or "cuda"


class MiniLMEncoder:
    def __init__(self, config: Optional[EncoderConfig] = None):
        self.config = config or EncoderConfig()
        self._model = None

    def _lazy_load(self):
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(MODEL_NAME, device=self.config.device)

    def encode(self, texts: List[str] | str) -> np.ndarray:
        """Returns an (n, 384) float32 array (or (384,) for a single string)."""
        self._lazy_load()
        single = isinstance(texts, str)
        inputs = [texts] if single else texts
        vectors = self._model.encode(
            inputs, convert_to_numpy=True, normalize_embeddings=True
        )
        return vectors[0] if single else vectors


if __name__ == "__main__":
    encoder = MiniLMEncoder()
    print("Configured device:", encoder.config.device)
    print("(Call .encode(...) to actually load the model and embed text.)")
