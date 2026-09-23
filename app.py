"""
app.py
======
A minimal, dependency-safe RAG chatbot.

- LLM: Qwen2.5-1.5B-Instruct (open source, Apache-2.0). Plain fp16/fp32
  load via transformers -- no auto-gptq, no bitsandbytes. Runs on CPU or
  GPU with nothing extra to compile. Change MODEL_NAME below for a bigger
  model if you have the VRAM (e.g. "Qwen/Qwen2.5-3B-Instruct" or
  "Qwen/Qwen2.5-7B-Instruct").
- Embeddings: all-MiniLM-L6-v2 (sentence-transformers), CPU.
- Vector store: FAISS, in-memory, rebuilt each session (no separate DB
  process, nothing to configure).
- Read: upload PDF / TXT / XLSX -> chunked, embedded, retrievable.
- Write: ask the model to produce a table and export it as a real .xlsx
  file via pandas/openpyxl.

Run:
    pip install -r requirements.txt
    python app.py
"""

import io
import re
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import gradio as gr
import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"   # open source, small, no GPTQ/bitsandbytes needed
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384
CHUNK_WORDS = 300
CHUNK_OVERLAP = 40
TOP_K = 4
MAX_NEW_TOKENS = 400

_device = "cuda" if torch.cuda.is_available() else "cpu"
_dtype = torch.float16 if _device == "cuda" else torch.float32

# ---------------------------------------------------------------------------
# Lazy-loaded models (loaded once, on first use)
# ---------------------------------------------------------------------------
_llm = None
_tokenizer = None
_embedder = None


def get_llm():
    global _llm, _tokenizer
    if _llm is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _llm = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=_dtype,
        ).to(_device)
    return _llm, _tokenizer


def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL_NAME, device="cpu")
    return _embedder


# ---------------------------------------------------------------------------
# In-memory vector store
# ---------------------------------------------------------------------------
index = faiss.IndexFlatIP(EMBED_DIM)
chunk_texts: list[str] = []
chunk_sources: list[str] = []


def chunk_text(text: str, words_per_chunk: int = CHUNK_WORDS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    words = text.split()
    if not words:
        return []
    chunks, start = [], 0
    while start < len(words):
        end = min(start + words_per_chunk, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = end - overlap
    return chunks


def read_any_file(path: str) -> str:
    suffix = Path(path).suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(path)
        return "\n".join((p.extract_text() or "") for p in reader.pages)
    elif suffix in (".txt", ".md", ".csv"):
        return Path(path).read_text(encoding="utf-8", errors="ignore")
    elif suffix in (".xlsx", ".xls"):
        df = pd.read_excel(path, sheet_name=None)  # all sheets
        parts = []
        for sheet_name, sheet_df in df.items():
            parts.append(f"[Sheet: {sheet_name}]\n{sheet_df.to_string(index=False)}")
        return "\n\n".join(parts)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")


def ingest_files(files) -> str:
    if not files:
        return "No files provided."
    embedder = get_embedder()
    total_chunks = 0
    for f in files:
        path = f.name if hasattr(f, "name") else f
        try:
            text = read_any_file(path)
        except Exception as e:
            return f"Failed to read {path}: {e}"
        chunks = chunk_text(text)
        if not chunks:
            continue
        vectors = embedder.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)
        index.add(vectors.astype(np.float32))
        chunk_texts.extend(chunks)
        chunk_sources.extend([Path(path).name] * len(chunks))
        total_chunks += len(chunks)
    return f"Ingested {len(files)} file(s), {total_chunks} chunks. Knowledge base now has {index.ntotal} chunks total."


def retrieve(query: str, k: int = TOP_K) -> list[str]:
    if index.ntotal == 0:
        return []
    embedder = get_embedder()
    qvec = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    k = min(k, index.ntotal)
    scores, idxs = index.search(qvec, k)
    results = []
    for score, idx in zip(scores[0], idxs[0]):
        if idx == -1:
            continue
        results.append(f"[{chunk_sources[idx]}] {chunk_texts[idx]}")
    return results


# ---------------------------------------------------------------------------
# LLM generation
# ---------------------------------------------------------------------------
def generate(user_message: str, context_chunks: list[str]) -> str:
    model, tokenizer = get_llm()

    system_prompt = (
        "You are a helpful assistant. If context from documents is provided, "
        "ground your answer in it and say so; otherwise answer from general knowledge."
    )
    if context_chunks:
        system_prompt += "\n\nContext:\n" + "\n---\n".join(context_chunks)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(_device)

    output = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    decoded = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return decoded.strip()


# ---------------------------------------------------------------------------
# Excel write-back: ask the model for a CSV-formatted table, parse it,
# save as a real .xlsx the user can download.
# ---------------------------------------------------------------------------
def generate_excel(description: str, context_chunks: list[str]):
    instruction = (
        f"{description}\n\n"
        "Respond with ONLY a CSV table (comma-separated, first row = column headers). "
        "No explanation, no markdown code fences, just the raw CSV."
    )
    raw = generate(instruction, context_chunks)

    # Strip accidental markdown fences if the model adds them anyway
    raw = re.sub(r"^```(csv)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()

    try:
        df = pd.read_csv(io.StringIO(raw))
    except Exception:
        # Fallback: single-column dump of raw lines if CSV parsing fails
        lines = [l for l in raw.splitlines() if l.strip()]
        df = pd.DataFrame({"output": lines})

    out_path = "/tmp/generated_table.xlsx"
    df.to_excel(out_path, index=False)
    return out_path, df.to_string(index=False)


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
def chat_fn(message, history):
    context = retrieve(message)
    return generate(message, context)


def ingest_fn(files):
    return ingest_files(files)


def excel_fn(description):
    if not description.strip():
        return None, "Please describe what table you want generated."
    context = retrieve(description)
    path, preview = generate_excel(description, context)
    return path, preview


with gr.Blocks(title="Simple RAG + Excel Assistant") as demo:
    gr.Markdown(
        "# Simple RAG Assistant\n"
        f"Open-source model: `{MODEL_NAME}` (no GPTQ, no bitsandbytes — plain load, runs on CPU or GPU).\n\n"
        "Upload documents (PDF / TXT / CSV / XLSX) below to ground answers in your own data, "
        "then chat normally. Use the second tab to generate a downloadable Excel file."
    )

    with gr.Tab("Chat"):
        gr.ChatInterface(fn=chat_fn)

    with gr.Tab("Upload documents"):
        file_input = gr.File(file_count="multiple", label="PDF / TXT / CSV / XLSX")
        ingest_btn = gr.Button("Ingest into knowledge base")
        ingest_status = gr.Textbox(label="Status", interactive=False)
        ingest_btn.click(ingest_fn, inputs=file_input, outputs=ingest_status)

    with gr.Tab("Generate Excel"):
        gr.Markdown(
            "Describe a table you want (optionally grounded in uploaded documents), "
            "and get back a downloadable .xlsx file."
        )
        excel_desc = gr.Textbox(label="Describe the table", placeholder="e.g. List the key figures from the uploaded report as a table with columns Metric and Value")
        excel_btn = gr.Button("Generate Excel")
        excel_file = gr.File(label="Download")
        excel_preview = gr.Textbox(label="Preview", interactive=False)
        excel_btn.click(excel_fn, inputs=excel_desc, outputs=[excel_file, excel_preview])


if __name__ == "__main__":
    demo.launch()
