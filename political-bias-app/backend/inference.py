"""Phase 1: multi-turn chat per persona, sequential CPU generation."""

from __future__ import annotations

import logging
import os
import time
from typing import Any

# Before importing transformers: avoid nested tqdm with our persona loop (misleading %).
os.environ.setdefault("TRANSFORMERS_NO_TQDM", "1")

import torch
from transformers import StoppingCriteria, StoppingCriteriaList

from session_store import append_turn, get_history

logger = logging.getLogger(__name__)

CPU = torch.device("cpu")

# Loaded at import (startup); never mutated at runtime.
SYSTEM_PROMPTS: dict[str, str] = {
    "democrat": (
        "You are a lifelong progressive Democrat. You are passionate, "
        "empathetic, and well-informed. You believe in expanding social "
        "programs, universal healthcare, addressing climate change urgently, "
        "protecting voting rights, and reducing income inequality through "
        "progressive taxation. You cite real Democratic policy positions "
        "and real political figures when relevant. You stay in character "
        "no matter what. Keep responses concise — 3 to 5 sentences max."
    ),
    "republican": (
        "You are a lifelong conservative Republican. You are confident, "
        "principled, and well-informed. You believe in free markets, limited "
        "government, strong national defense, border security, second amendment "
        "rights, low taxes, and traditional values. You cite real Republican "
        "policy positions and real political figures when relevant. You stay "
        "in character no matter what. Keep responses concise — 3 to 5 sentences max."
    ),
    "centrist": (
        "You are a pragmatic political independent and centrist. You are "
        "frustrated by partisan gridlock and believe both parties have valid "
        "points and serious flaws. You prioritize evidence-based policy over "
        "ideology. You challenge left and right talking points equally and "
        "look for common ground. Keep responses concise — 3 to 5 sentences max."
    ),
}

# Shorter cap helps slow CPUs finish before REQUEST_TIMEOUT_SECONDS (still override via env).
MAX_NEW_TOKENS = int(os.environ.get("CHAT_MAX_NEW_TOKENS", "128"))
# Greedy decoding is faster on CPU than sampling (set CHAT_GREEDY=0 to restore sampling).
_CHAT_GREEDY = os.environ.get("CHAT_GREEDY", "1").strip().lower() in ("1", "true", "yes")


class WallClockTimeout(StoppingCriteria):
    def __init__(self, start: float, limit_seconds: float, flag: dict[str, bool]) -> None:
        self.start = start
        self.limit = limit_seconds
        self.flag = flag

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if (time.perf_counter() - self.start) > self.limit:
            self.flag["hit"] = True
            return True
        return False


def _build_messages(persona: str, history: list[dict[str, str]], user_message: str) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPTS[persona]}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_message})
    return messages


@torch.no_grad()
def _generate_one(
    store: dict[str, Any],
    messages: list[dict[str, str]],
    per_persona_timeout_s: float,
) -> tuple[str, float]:
    tokenizer = store["tokenizer"]
    model = store["model"]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(CPU) for k, v in inputs.items()}
    prompt_len = int(inputs["input_ids"].shape[1])
    logger.info(
        "generate start prompt_tokens=%s max_new_tokens=%s use_cache=False",
        prompt_len,
        MAX_NEW_TOKENS,
    )

    start = time.perf_counter()
    timeout_flag: dict[str, bool] = {"hit": False}
    stopping = StoppingCriteriaList([WallClockTimeout(start, per_persona_timeout_s, timeout_flag)])

    gen_kw: dict[str, Any] = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "pad_token_id": tokenizer.eos_token_id,
        "return_dict_in_generate": True,
        "stopping_criteria": stopping,
        "use_cache": False,
    }
    if _CHAT_GREEDY:
        gen_kw["do_sample"] = False
    else:
        gen_kw["do_sample"] = True
        gen_kw["temperature"] = 0.7
        gen_kw["top_p"] = 0.9

    out = model.generate(**inputs, **gen_kw)
    seq = out.sequences
    elapsed = time.perf_counter() - start

    if timeout_flag["hit"]:
        msg = (
            f"Timed out after {int(per_persona_timeout_s)}s (REQUEST_TIMEOUT_SECONDS). "
            f"Tuning: lower CHAT_MAX_NEW_TOKENS (now {MAX_NEW_TOKENS}), set CHAT_GREEDY=1 for "
            "faster greedy decode, or raise REQUEST_TIMEOUT_SECONDS."
        )
        return msg, round(elapsed, 2)

    new_tokens = seq[0, prompt_len:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    logger.info(
        "generate done new_tokens=%s elapsed_s=%.2f",
        int(new_tokens.shape[0]),
        elapsed,
    )
    return text or "(empty response)", round(elapsed, 2)


def run_chat_turn(
    store: dict[str, Any],
    session_id: str,
    user_message: str,
    per_persona_timeout_s: float,
) -> dict[str, dict[str, Any]]:
    """Sequential: democrat → republican → centrist. Updates session_store."""
    order = ["democrat", "republican", "centrist"]
    out: dict[str, dict[str, Any]] = {}

    for persona in order:
        history = get_history(session_id, persona)
        messages = _build_messages(persona, history, user_message)
        logger.info(
            "chat turn persona=%s prior_user_assistant_pairs=%s",
            persona,
            len(history) // 2,
        )
        t0 = time.perf_counter()
        try:
            reply, infer_s = _generate_one(store, messages, per_persona_timeout_s)
        except Exception as exc:
            logger.exception("generate failed persona=%s", persona)
            reply = f"Error: {exc}"
            infer_s = round(time.perf_counter() - t0, 2)
        append_turn(session_id, persona, user_message, reply)
        out[persona] = {
            "response": reply,
            "inference_time_seconds": float(infer_s),
        }
        logger.info("chat persona=%s inference_s=%.2f", persona, infer_s)

    return out
