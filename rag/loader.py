"""
rag/loader.py
=============
Loads raw text out of PDF or TXT files for the RAG pipeline.
"""

from pathlib import Path


def load_txt(path: str) -> str:
    return Path(path).read_text(encoding="utf-8", errors="ignore")


def load_pdf(path: str) -> str:
    # Deferred import: keeps this module importable without pypdf installed.
    from pypdf import PdfReader

    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def load_document(path: str) -> str:
    suffix = Path(path).suffix.lower()
    if suffix == ".pdf":
        return load_pdf(path)
    elif suffix in (".txt", ".md"):
        return load_txt(path)
    else:
        raise ValueError(f"Unsupported document type: {suffix}")
