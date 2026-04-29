import logging
import time
from typing import Any

import torch
import torch.nn.functional as F
from transformers import StoppingCriteria, StoppingCriteriaList

logger = logging.getLogger(__name__)

CPU = torch.device("cpu")

PROMPTS: dict[str, str] = {
    "democrat": (
        "You are a progressive Democrat commentator. Analyze the following "
        "news from a liberal perspective. Focus on social equity, climate policy, "
        "healthcare access, and the role of government. Be specific and analytical."
    ),
    "republican": (
        "You are a conservative Republican commentator. Analyze the "
        "following news from a conservative perspective. Focus on free markets, "
        "individual liberty, border security, and limited government. Be specific "
        "and analytical."
    ),
    "centrist": (
        "You are a nonpartisan political analyst. Analyze the following "
        "news objectively. Present the strongest arguments from both left and right, "
        "then give a balanced assessment. Avoid partisan framing."
    ),
}

TIMEOUT_MESSAGE = (
    "Inference exceeded the time limit for this persona. Try a shorter headline or article."
)
MAX_NEW_TOKENS = 256


class WallClockTimeout(StoppingCriteria):
    """Cooperative stop when wall-clock exceeds limit; sets flag when triggered."""

    def __init__(self, start: float, limit_seconds: float, flag: dict[str, bool]) -> None:
        self.start = start
        self.limit = limit_seconds
        self.flag = flag

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if (time.perf_counter() - self.start) > self.limit:
            self.flag["hit"] = True
            return True
        return False


def _build_chat_prompt(tokenizer, system_prompt: str, news: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"News to analyze:\n{news}"},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def _mean_chosen_token_confidence(
    scores: tuple[torch.Tensor, ...], sequences: torch.Tensor, prompt_len: int
) -> float:
    if not scores:
        return 0.0
    gen_ids = sequences[0, prompt_len:]
    n = min(len(scores), gen_ids.shape[0])
    if n == 0:
        return 0.0
    total = 0.0
    for step in range(n):
        logits = scores[step][0]
        probs = F.softmax(logits.float(), dim=-1)
        tok = int(gen_ids[step].item())
        total += float(probs[tok].clamp(0.0, 1.0).item())
    return total / n


@torch.no_grad()
def _run_one_persona(
    store: dict[str, Any],
    persona: str,
    news: str,
    per_persona_timeout_s: float,
) -> dict[str, Any]:
    model = store["model"]
    tokenizer = store["tokenizer"]
    adapter_names: dict[str, str] = store["adapter_names"]

    model.set_adapter(adapter_names[persona])
    prompt = _build_chat_prompt(tokenizer, PROMPTS[persona], news)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=3072)
    inputs = {k: v.to(CPU) for k, v in inputs.items()}
    prompt_len = int(inputs["input_ids"].shape[1])

    start = time.perf_counter()
    timeout_flag: dict[str, bool] = {"hit": False}
    stopping = StoppingCriteriaList([WallClockTimeout(start, per_persona_timeout_s, timeout_flag)])

    out = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,
        output_scores=True,
        stopping_criteria=stopping,
    )
    seq = out.sequences
    scores = out.scores or ()
    elapsed = time.perf_counter() - start

    if timeout_flag["hit"]:
        logger.warning(
            "Persona %s stopped by time limit (%.2fs / %.2fs)",
            persona,
            elapsed,
            per_persona_timeout_s,
        )
        return {
            "response": TIMEOUT_MESSAGE,
            "confidence": 0.0,
            "inference_seconds": round(elapsed, 2),
            "timed_out": True,
        }

    new_tokens = seq[0, prompt_len:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    conf = _mean_chosen_token_confidence(scores, seq, prompt_len)
    return {
        "response": text,
        "confidence": round(min(1.0, max(0.0, conf)), 4),
        "inference_seconds": round(elapsed, 2),
        "timed_out": False,
    }


def run_all_personas_sequential(
    store: dict[str, Any],
    news: str,
    per_persona_timeout_s: float,
) -> dict[str, dict[str, Any]]:
    order = ["democrat", "republican", "centrist"]
    out: dict[str, dict[str, Any]] = {}
    for persona in order:
        t0 = time.perf_counter()
        row = _run_one_persona(store, persona, news, per_persona_timeout_s)
        dt = time.perf_counter() - t0
        logger.info(
            "inference persona=%s wall_seconds=%.2f reported=%s timed_out=%s",
            persona,
            dt,
            row.get("inference_seconds"),
            row.get("timed_out"),
        )
        out[persona] = {
            "response": row["response"],
            "confidence": row["confidence"],
            "inference_seconds": row.get("inference_seconds", round(dt, 2)),
        }
    return out
