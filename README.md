# TOC-chat — Simple RAG Assistant

A minimal, dependency-safe RAG chatbot in two files.

## Architecture

![TOC-chat architecture diagram](docs/architecture.svg)

## Features

- **LLM**: `Qwen/Qwen2.5-1.5B-Instruct` — open source (Apache 2.0), plain
  `transformers` load (fp16 on GPU / fp32 on CPU). No GPTQ, no
  `auto-gptq`, no `bitsandbytes` — nothing that needs a compiled wheel or
  build toolchain. Change `MODEL_NAME` in `app.py` for a bigger model
  (e.g. `Qwen/Qwen2.5-3B-Instruct` or `Qwen/Qwen2.5-7B-Instruct`) if you
  have the VRAM.
- **Embeddings**: `all-MiniLM-L6-v2` (sentence-transformers), CPU.
- **Vector store**: plain NumPy array with cosine similarity — no FAISS
  dependency, since `faiss-cpu` has no reliable wheel across every
  Python version Colab ships with.
- **Read**: upload PDF / TXT / CSV / XLSX — chunked, embedded, retrievable
  for grounded answers.
- **Write**: ask for a table (optionally grounded in your uploaded docs)
  and download it as a real `.xlsx` file.
- **Memory**: every chat turn is saved to `chat_memory.json` and the most
  recent exchanges are fed back into the model on each new message, so it
  stays consistent across a conversation and across restarts. A "Clear
  conversation memory" button resets it.

## Setup

```bash
pip install -r requirements.txt
python app.py
```

Opens a Gradio UI with three tabs: **Chat**, **Upload documents**, and
**Generate Excel**.

## Why this structure

Earlier versions of this repo used GPTQ checkpoints (`auto-gptq`) and a
multi-module architecture. `auto-gptq` has no reliable prebuilt wheels on
most platforms and regularly fails to build from source. This version
trades some of that structure for reliability: two files, all
dependencies with broad prebuilt-wheel coverage, one small open-source
model that runs on CPU or GPU with no special quantization library.
