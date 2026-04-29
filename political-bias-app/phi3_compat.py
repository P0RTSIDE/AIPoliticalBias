"""
Phi-3 + Transformers 5.x: hub config may use rope_scaling['rope_type'] while
remote modeling_phi3.py still reads rope_scaling['type']. Patch before load.
"""
from __future__ import annotations

from typing import Any

import torch


def cpu_supports_bfloat16() -> bool:
    """torch.cpu.is_bf16_supported exists on newer builds; some installs omit it."""
    fn = getattr(torch.cpu, "is_bf16_supported", None)
    if callable(fn):
        try:
            return bool(fn())
        except Exception:
            return False
    return False


def patch_phi3_rope_config(config: Any) -> Any:
    rs = getattr(config, "rope_scaling", None)
    if not isinstance(rs, dict):
        return config
    if "type" not in rs and "rope_type" in rs:
        patched = dict(rs)
        patched["type"] = patched["rope_type"]
        config.rope_scaling = patched
    return config
