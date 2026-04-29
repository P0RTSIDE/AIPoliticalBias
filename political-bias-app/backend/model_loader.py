import os
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

_APP = Path(__file__).resolve().parents[1]
if str(_APP) not in sys.path:
    sys.path.insert(0, str(_APP))
from phi3_compat import cpu_supports_bfloat16, patch_phi3_rope_config


PERSONAS = ["democrat", "republican", "centrist"]

DEFAULT_BASE = "microsoft/Phi-3-mini-4k-instruct"


def load_models_dict(adapters_root: Path, base_model: str | None = None) -> dict:
    """
    Load base Phi-3 once on CPU with bfloat16 (or float32 if BF16 unsupported),
    attach three LoRA adapters. Returns a dict for FastAPI to keep in memory.
    """
    selected = (base_model or os.environ.get("BASE_MODEL") or DEFAULT_BASE).strip()
    dtype = torch.bfloat16 if cpu_supports_bfloat16() else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(
        selected, use_fast=True, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(selected, trust_remote_code=True)
    patch_phi3_rope_config(config)
    base = AutoModelForCausalLM.from_pretrained(
        selected,
        config=config,
        trust_remote_code=True,
        device_map="cpu",
        torch_dtype=dtype,
        attn_implementation="eager",
    )

    first = adapters_root / PERSONAS[0] / "adapter"
    model = PeftModel.from_pretrained(base, str(first), adapter_name=PERSONAS[0])
    adapter_names: dict[str, str] = {PERSONAS[0]: PERSONAS[0]}

    for persona in PERSONAS[1:]:
        path = adapters_root / persona / "adapter"
        model.load_adapter(str(path), adapter_name=persona)
        adapter_names[persona] = persona

    return {
        "tokenizer": tokenizer,
        "model": model,
        "adapter_names": adapter_names,
        "device": "cpu",
        "dtype": str(dtype),
        "base_model": selected,
    }
