"""Phase 1: load Phi-3-mini once on CPU, float32, eager attention."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_NO_TQDM", "1")

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

_APP = Path(__file__).resolve().parents[1]
if str(_APP) not in sys.path:
    sys.path.insert(0, str(_APP))
from phi3_compat import patch_phi3_rope_config

DEFAULT_BASE = "microsoft/Phi-3-mini-4k-instruct"


def load_model_store(base_model: str | None = None) -> dict:
    selected = (base_model or os.environ.get("BASE_MODEL") or DEFAULT_BASE).strip()

    tokenizer = AutoTokenizer.from_pretrained(
        selected,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(selected, trust_remote_code=True)
    patch_phi3_rope_config(config)

    model = AutoModelForCausalLM.from_pretrained(
        selected,
        config=config,
        dtype=torch.float32,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.to(torch.device("cpu"))
    model.eval()
    # Remote Phi-3 hub code + Transformers 5.x: DynamicCache API mismatch
    # ("seen_tokens"). Disabling cache avoids that path for generate().
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.use_cache = False

    return {
        "tokenizer": tokenizer,
        "model": model,
        "device": "cpu",
        "dtype": "torch.float32",
        "base_model": selected,
    }
