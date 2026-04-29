import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from sklearn.metrics import accuracy_score, classification_report, f1_score
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

_APP = Path(__file__).resolve().parents[1]
if str(_APP) not in sys.path:
    sys.path.insert(0, str(_APP))
from phi3_compat import cpu_supports_bfloat16, patch_phi3_rope_config


PERSONAS = ["democrat", "republican", "centrist"]
DEFAULT_BASE = "microsoft/Phi-3-mini-4k-instruct"


def load_model_cpu_bf16(base_model: str, adapter_path: Path) -> tuple[PeftModel, AutoTokenizer]:
    dtype = torch.bfloat16 if cpu_supports_bfloat16() else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(
        base_model, use_fast=True, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    config = AutoConfig.from_pretrained(base_model, trust_remote_code=True)
    patch_phi3_rope_config(config)
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        config=config,
        trust_remote_code=True,
        device_map="cpu",
        torch_dtype=dtype,
        attn_implementation="eager",
    )
    model = PeftModel.from_pretrained(base, str(adapter_path))
    model.eval()
    return model, tokenizer


@torch.no_grad()
def avg_perplexity(model: PeftModel, tokenizer: AutoTokenizer, texts: list[str], max_length: int = 512) -> float:
    losses: list[float] = []
    cpu = torch.device("cpu")
    for text in texts:
        toks = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        toks = {k: v.to(cpu) for k, v in toks.items()}
        out = model(**toks, labels=toks["input_ids"])
        losses.append(out.loss.item())
    return float(math.exp(np.mean(losses))) if losses else float("inf")


@torch.no_grad()
def predict_label(
    models: dict[str, PeftModel],
    tokenizers: dict[str, AutoTokenizer],
    text: str,
    max_length: int = 512,
) -> str:
    ppl: dict[str, float] = {}
    cpu = torch.device("cpu")
    for persona in PERSONAS:
        model = models[persona]
        tokenizer = tokenizers[persona]
        toks = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        toks = {k: v.to(cpu) for k, v in toks.items()}
        out = model(**toks, labels=toks["input_ids"])
        ppl[persona] = math.exp(out.loss.item())
    return min(ppl, key=ppl.get)


@torch.no_grad()
def sample_generation(
    model: PeftModel,
    tokenizer: AutoTokenizer,
    system_prompt: str,
    news: str,
    max_new_tokens: int = 128,
) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"News:\n{news}\n\nBrief analysis:"},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(torch.device("cpu")) for k, v in inputs.items()}
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
    )
    gen = out[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


def evaluate(
    processed_dir: Path,
    adapters_dir: Path,
    base_model: str,
    output_json: Path,
    num_samples: int = 2,
) -> None:
    test_df = pd.read_parquet(processed_dir / "test.parquet")
    models: dict[str, PeftModel] = {}
    tokenizers: dict[str, AutoTokenizer] = {}
    for persona in PERSONAS:
        adapter_path = adapters_dir / persona / "adapter"
        models[persona], tokenizers[persona] = load_model_cpu_bf16(base_model, adapter_path)

    metrics: dict = {}
    for persona in PERSONAS:
        texts = test_df[test_df["label"] == persona]["text"].tolist()
        metrics[persona] = {
            "perplexity_on_class": avg_perplexity(models[persona], tokenizers[persona], texts[:200]),
        }

    y_true = test_df["label"].tolist()
    y_pred = [predict_label(models, tokenizers, t) for t in test_df["text"].tolist()]
    metrics["overall"] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "report": classification_report(y_true, y_pred, output_dict=True),
    }

    system = (
        "You are a nonpartisan analyst. Summarize the news in one short paragraph."
    )
    sample_outputs: dict[str, list[str]] = {p: [] for p in PERSONAS}
    for persona in PERSONAS:
        subset = test_df[test_df["label"] == persona]["text"].head(num_samples)
        for text in subset:
            sample_outputs[persona].append(
                sample_generation(models[persona], tokenizers[persona], system, str(text))
            )
    metrics["sample_outputs"] = sample_outputs

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", type=Path, default=Path("artifacts/processed"))
    parser.add_argument("--adapters_dir", type=Path, default=Path("artifacts/adapters"))
    parser.add_argument("--base_model", type=str, default=DEFAULT_BASE)
    parser.add_argument("--output_json", type=Path, default=Path("artifacts/metrics.json"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args.processed_dir, args.adapters_dir, args.base_model, args.output_json)
