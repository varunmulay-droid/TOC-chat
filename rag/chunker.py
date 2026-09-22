"""
rag/chunker.py
==============
Splits document text into ~500-800 token chunks (approximated by word
count, ~0.75 tokens/word average for English), matching the T4 memory
strategy's target chunk size. Word-boundary splitting with overlap keeps
context from being severed mid-sentence too often.
"""

from typing import List

WORDS_PER_CHUNK = 550    # approx. targets ~500-800 tokens
OVERLAP_WORDS = 50


def chunk_text(text: str, words_per_chunk: int = WORDS_PER_CHUNK, overlap: int = OVERLAP_WORDS) -> List[str]:
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + words_per_chunk, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = end - overlap  # step back for overlap
    return chunks


if __name__ == "__main__":
    sample = "word " * 1200
    result = chunk_text(sample)
    print(f"Produced {len(result)} chunks from 1200 words")
    print("First chunk word count:", len(result[0].split()))
