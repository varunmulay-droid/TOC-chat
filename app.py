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
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import gradio as gr
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
MEMORY_FILE = Path("chat_memory.json")   # plain JSON, no DB server/driver to break
MEMORY_TURNS_IN_PROMPT = 6                # how many recent exchanges to feed back as context

import threading

_device = "cuda" if torch.cuda.is_available() else "cpu"
_dtype = torch.float16 if _device == "cuda" else torch.float32

# Max prompt tokens allowed before we truncate context/memory. Prevents a
# single very long prompt (big RAG context + long memory history) from
# causing a huge quadratic attention allocation.
MAX_PROMPT_TOKENS = 2048

# ---------------------------------------------------------------------------
# Lazy-loaded models (loaded once, on first use)
# ---------------------------------------------------------------------------
_llm = None
_tokenizer = None
_embedder = None

# Gradio runs each tab's callback on its own worker thread. Without a lock,
# two requests arriving close together (e.g. a chat message and an Excel
# generation) can both see the model as unloaded and each load a FULL
# separate copy onto the GPU at the same time -- doubling memory before
# generation even starts. This lock serializes both loading AND generation,
# since a single T4/consumer GPU can't safely run two generate() calls
# concurrently anyway.
_gpu_lock = threading.Lock()


def get_llm():
    global _llm, _tokenizer
    if _llm is None:
        with _gpu_lock:
            if _llm is None:  # re-check after acquiring the lock
                _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
                _llm = AutoModelForCausalLM.from_pretrained(
                    MODEL_NAME,
                    dtype=_dtype,               # transformers>=4.5x renamed torch_dtype -> dtype;
                                                  # the old kwarg is silently ignored on newer
                                                  # versions, which loads fp32 and doubles memory
                    low_cpu_mem_usage=True,
                ).to(_device)
    return _llm, _tokenizer


def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL_NAME, device="cpu")
    return _embedder


# ---------------------------------------------------------------------------
# In-memory vector store (plain NumPy — no FAISS dependency to break on
# platform/Python-version mismatches). Embeddings are L2-normalized, so a
# dot product is equivalent to cosine similarity.
# ---------------------------------------------------------------------------
chunk_vectors: np.ndarray | None = None   # shape (n_chunks, EMBED_DIM)
chunk_texts: list[str] = []
chunk_sources: list[str] = []


def add_vectors(vectors: np.ndarray):
    global chunk_vectors
    vectors = vectors.astype(np.float32)
    if chunk_vectors is None:
        chunk_vectors = vectors
    else:
        chunk_vectors = np.vstack([chunk_vectors, vectors])


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
        add_vectors(vectors)
        chunk_texts.extend(chunks)
        chunk_sources.extend([Path(path).name] * len(chunks))
        total_chunks += len(chunks)
    return f"Ingested {len(files)} file(s), {total_chunks} chunks. Knowledge base now has {len(chunk_texts)} chunks total."


def retrieve(query: str, k: int = TOP_K) -> list[str]:
    if chunk_vectors is None or len(chunk_texts) == 0:
        return []
    embedder = get_embedder()
    qvec = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)[0]
    scores = chunk_vectors @ qvec  # cosine similarity, since both sides are normalized
    k = min(k, len(chunk_texts))
    top_idxs = np.argsort(-scores)[:k]
    return [f"[{chunk_sources[i]}] {chunk_texts[i]}" for i in top_idxs]


# ---------------------------------------------------------------------------
# Conversation memory — plain JSON file, no DB server/driver required.
# Persists across restarts (survives a Colab session as long as the file
# system does); each turn is appended, and the most recent N turns are fed
# back to the model as extra context so it "remembers" the conversation.
# ---------------------------------------------------------------------------
def load_memory() -> list[dict]:
    if not MEMORY_FILE.exists():
        return []
    try:
        return json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []  # corrupt or unreadable file -- start fresh rather than crash


def save_turn(user_text: str, assistant_text: str):
    history = load_memory()
    history.append({
        "timestamp": time.time(),
        "user": user_text,
        "assistant": assistant_text,
    })
    MEMORY_FILE.write_text(json.dumps(history, indent=2), encoding="utf-8")


def recent_memory_as_messages(n: int = MEMORY_TURNS_IN_PROMPT) -> list[dict]:
    """Return the last n turns as chat-template-ready message dicts."""
    history = load_memory()[-n:]
    messages = []
    for turn in history:
        messages.append({"role": "user", "content": turn["user"]})
        messages.append({"role": "assistant", "content": turn["assistant"]})
    return messages


def clear_memory():
    if MEMORY_FILE.exists():
        MEMORY_FILE.unlink()
    return "Conversation memory cleared."


# ---------------------------------------------------------------------------
# LLM generation
# ---------------------------------------------------------------------------
def generate(user_message: str, context_chunks: list[str], use_memory: bool = True) -> str:
    model, tokenizer = get_llm()

    system_prompt = (
        "You are a helpful assistant. If context from documents is provided, "
        "ground your answer in it and say so; otherwise answer from general knowledge. "
        "You also have access to recent conversation history -- use it to stay "
        "consistent with what's already been discussed."
    )
    if context_chunks:
        system_prompt += "\n\nContext:\n" + "\n---\n".join(context_chunks)

    system_msg = {"role": "system", "content": system_prompt}
    user_msg = {"role": "user", "content": user_message}
    memory_msgs = recent_memory_as_messages() if use_memory else []

    def build_and_count(mem_msgs):
        msgs = [system_msg] + mem_msgs + [user_msg]
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        token_count = len(tokenizer(text, return_tensors="pt")["input_ids"][0])
        return text, token_count

    text, token_count = build_and_count(memory_msgs)
    # If the prompt (system + memory + current message) exceeds the budget,
    # drop the OLDEST memory turns first (two messages at a time -- one
    # user/assistant pair) until it fits. This is what actually caps the
    # quadratic attention-memory blowup, rather than just capping turn count,
    # since a single earlier response full of code can be hundreds of tokens.
    while token_count > MAX_PROMPT_TOKENS and len(memory_msgs) >= 2:
        memory_msgs = memory_msgs[2:]
        text, token_count = build_and_count(memory_msgs)

    # If it's STILL too big even with zero memory (e.g. huge RAG context),
    # truncate the tokenized input directly as a last resort.
    inputs = tokenizer(text, return_tensors="pt")
    if inputs["input_ids"].shape[1] > MAX_PROMPT_TOKENS:
        inputs["input_ids"] = inputs["input_ids"][:, -MAX_PROMPT_TOKENS:]
        inputs["attention_mask"] = inputs["attention_mask"][:, -MAX_PROMPT_TOKENS:]
    inputs = inputs.to(_device)

    # Serialize GPU access: only one generate() call runs at a time, so two
    # concurrent Gradio requests (e.g. chat + Excel generation) can't both
    # try to hold model activations on the GPU simultaneously.
    with _gpu_lock:
        try:
            output = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            raise RuntimeError(
                "GPU ran out of memory generating this response. Try a shorter message, "
                "fewer uploaded documents, or restart the runtime to clear GPU memory."
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
    raw = generate(instruction, context_chunks, use_memory=False)

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
    response = generate(message, context)
    save_turn(message, response)
    return response


def ingest_fn(files):
    return ingest_files(files)


def excel_fn(description):
    if not description.strip():
        return None, "Please describe what table you want generated."
    context = retrieve(description)
    path, preview = generate_excel(description, context)
    return path, preview


with gr.Blocks(title="Simple RAG + Excel Assistant", default_concurrency_limit=1) as demo:
    gr.Markdown(
        "# Simple RAG Assistant\n"
        f"Open-source model: `{MODEL_NAME}` (no GPTQ, no bitsandbytes — plain load, runs on CPU or GPU).\n\n"
        "Upload documents (PDF / TXT / CSV / XLSX) below to ground answers in your own data, "
        "then chat normally. Use the second tab to generate a downloadable Excel file."
    )

    with gr.Tab("Chat"):
        gr.ChatInterface(fn=chat_fn)
        with gr.Row():
            clear_mem_btn = gr.Button("Clear conversation memory")
            clear_mem_status = gr.Textbox(label="Status", interactive=False, scale=2)
        clear_mem_btn.click(clear_memory, outputs=clear_mem_status)

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
