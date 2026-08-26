"""Model loading helpers.

One entry point, `load_model`, returns a `LoadedModel` with the pieces every
other module needs: the HF model, tokenizer, device, the list of transformer
blocks (residual-stream write points), and the sizes.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

# Models small enough to run on the Mac (MPS / CPU) for pipeline tests.
TINY_MODELS = {
    "qwen0.5": "Qwen/Qwen2.5-0.5B-Instruct",
    "gpt2": "gpt2",
}


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_blocks(model: PreTrainedModel) -> nn.ModuleList:
    """Return the ModuleList of decoder blocks (the residual stream is read/written at their outputs).

    Covers Llama / Qwen2 / Qwen3 / Mistral / Gemma (`model.model.layers`) and GPT-2 (`model.transformer.h`).
    """
    for path in ("model.layers", "model.language_model.layers", "language_model.layers",
                 "transformer.h", "model.decoder.layers", "gpt_neox.layers"):
        obj = model
        ok = True
        for attr in path.split("."):
            if not hasattr(obj, attr):
                ok = False
                break
            obj = getattr(obj, attr)
        if ok and isinstance(obj, nn.ModuleList):
            return obj
    raise ValueError(f"Cannot find decoder blocks on {type(model).__name__}; add its path to get_blocks()")


@dataclass
class LoadedModel:
    name: str
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    device: str
    blocks: nn.ModuleList
    n_layers: int
    d_model: int
    dtype: torch.dtype

    def chat(self, user: str, system: str | None = None, assistant: str | None = None,
             add_generation_prompt: bool = True) -> str:
        """Render a chat-formatted prompt string. If the tokenizer has no chat template (gpt2), fall back to plain text."""
        if getattr(self.tokenizer, "chat_template", None):
            msgs = []
            if system:
                msgs.append({"role": "system", "content": system})
            msgs.append({"role": "user", "content": user})
            if assistant is not None:
                msgs.append({"role": "assistant", "content": assistant})
            return self.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=add_generation_prompt and assistant is None
            )
        text = (f"{system}\n\n" if system else "") + user
        if assistant is not None:
            text += "\n" + assistant
        return text


def load_model(name: str, dtype: torch.dtype | None = None, device: str | None = None,
               attn_implementation: str | None = None, seed: int | None = 0,
               load_in_4bit: bool = False, adapter: str | None = None) -> LoadedModel:
    """Load a causal LM + tokenizer ready for batched activation capture.

    dtype defaults to bf16 on CUDA, fp16 on MPS, fp32 on CPU. Left padding is set so
    that "last token" positions line up in a batch. `load_in_4bit` uses nf4 with bf16
    compute (for 70B on one 80GB card); `adapter` merges a PEFT LoRA for probing the
    fine-tuned model's residual stream.
    """
    name = TINY_MODELS.get(name, name)
    device = device or pick_device()
    if dtype is None:
        dtype = {"cuda": torch.bfloat16, "mps": torch.float16, "cpu": torch.float32}[device]
    if seed is not None:
        seed_all(seed)

    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    kwargs = dict(torch_dtype=dtype)
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
        kwargs["device_map"] = "auto"
    elif device == "cuda":
        kwargs["device_map"] = "cuda"
    model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
    if adapter is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    if device != "cuda" and not load_in_4bit:
        model.to(device)
    model.eval()

    blocks = get_blocks(model)
    cfg = model.config
    d_model = getattr(cfg, "hidden_size", None) or getattr(cfg, "n_embd")
    return LoadedModel(
        name=name, model=model, tokenizer=tok, device=device, blocks=blocks,
        n_layers=len(blocks), d_model=int(d_model), dtype=dtype,
    )


def vram_report() -> str:
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        return f"{torch.cuda.get_device_name(0)} | VRAM free {free / 1e9:.1f} / {total / 1e9:.1f} GB"
    if torch.backends.mps.is_available():
        return "Apple MPS (unified memory)"
    return f"CPU only ({os.cpu_count()} cores)"
