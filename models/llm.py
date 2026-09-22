"""
models/llm.py
==============
LLM loading + generation, wrapped so the model SIZE is a config choice,
not a hardcoded assumption.

Why not just hardcode Qwen2.5-7B-Instruct-GPTQ-Int4:
  - On free Colab T4 (16GB) and especially HF ZeroGPU (cold-start budget
    matters), a 7B 4-bit model's load time + KV cache + concurrent MiniLM
    encoder can blow the memory/time budget once you add RAG context and
    conversation history to the prompt.
  - Defaulting to 3B keeps the demo reliably working; 7B remains available
    as an opt-in for users running on Colab Pro / a bigger GPU.

Set TOC_GPT_MODEL env var to override, e.g.:
    TOC_GPT_MODEL=Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4
"""

import os
from dataclasses import dataclass
from typing import Optional

DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4"
LARGE_MODEL = "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4"   # opt-in, needs more VRAM / time


@dataclass
class LLMConfig:
    model_name: str = os.environ.get("TOC_GPT_MODEL", DEFAULT_MODEL)
    max_new_tokens: int = 384          # keep within the 256-512 T4-friendly budget
    temperature: float = 0.7
    top_p: float = 0.9
    device_map: str = "auto"


class QwenLLM:
    """Lazy-loading wrapper. The model is NOT loaded at import time --
    only on first .generate() call -- so importing this module (e.g. for
    unit-testing the TOC layer) never requires a GPU or downloads weights.
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self._model = None
        self._tokenizer = None

    def _lazy_load(self):
        if self._model is not None:
            return
        # Deferred import: keeps this module importable without torch/transformers
        # installed, which matters for quick unit tests of the TOC layer.
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        print(f"[llm] Loading {self.config.model_name} ...")
        self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            device_map=self.config.device_map,
            torch_dtype=torch.float16,
        )
        print("[llm] Model loaded.")

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        self._lazy_load()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)

        output = self._model.generate(
            **inputs,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            do_sample=True,
        )
        decoded = self._tokenizer.decode(
            output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        return decoded.strip()

    def clear_cache(self):
        """Call between heavy requests on memory-constrained GPUs (T4 / ZeroGPU)."""
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    # Import-only smoke test -- does not require a GPU.
    llm = QwenLLM()
    print("Configured model:", llm.config.model_name)
    print("(Call .generate() to actually load weights and run inference.)")
