import argparse
import inspect
import sys
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from tqdm.auto import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

_APP = Path(__file__).resolve().parents[1]
if str(_APP) not in sys.path:
    sys.path.insert(0, str(_APP))
from phi3_compat import cpu_supports_bfloat16, patch_phi3_rope_config


PERSONAS = ["democrat", "republican", "centrist"]

DEFAULT_BASE = "microsoft/Phi-3-mini-4k-instruct"


def build_dataset(df: pd.DataFrame) -> Dataset:
    return Dataset.from_dict({"text": df["text"].tolist()})


def tokenize_dataset(dataset: Dataset, tokenizer: AutoTokenizer, max_length: int) -> Dataset:
    def _tok(batch):
        out = tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        out["labels"] = out["input_ids"].copy()
        return out

    return dataset.map(_tok, batched=True, remove_columns=["text"])


def load_base_model(base_model: str) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(
        base_model, use_fast=True, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(base_model, trust_remote_code=True)
    patch_phi3_rope_config(config)
    dtype = torch.bfloat16 if cpu_supports_bfloat16() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        config=config,
        trust_remote_code=True,
        device_map="cpu",
        torch_dtype=dtype,
        attn_implementation="eager",
    )
    return model, tokenizer


def train_persona(
    persona: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    base_model: str,
    output_root: Path,
    epochs: int,
    lr: float,
    max_length: int,
) -> None:
    persona_train = train_df[train_df["label"] == persona]
    persona_val = val_df[val_df["label"] == persona]
    if persona_train.empty or persona_val.empty:
        raise ValueError(f"Persona '{persona}' has no training/validation rows.")

    model, tokenizer = load_base_model(base_model)
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(model, peft_config)

    train_ds = tokenize_dataset(build_dataset(persona_train), tokenizer, max_length)
    val_ds = tokenize_dataset(build_dataset(persona_val), tokenizer, max_length)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    out_dir = output_root / persona
    out_dir.mkdir(parents=True, exist_ok=True)

    ta_params = inspect.signature(TrainingArguments).parameters
    eval_kw: dict = {}
    if "eval_strategy" in ta_params:
        eval_kw["eval_strategy"] = "epoch"
    else:
        eval_kw["evaluation_strategy"] = "epoch"

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=epochs,
        learning_rate=lr,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        fp16=False,
        bf16=False,
        report_to="none",
        dataloader_pin_memory=False,
        **eval_kw,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
    )
    trainer.train()
    adapter_dir = out_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", type=Path, default=Path("artifacts/processed"))
    parser.add_argument("--base_model", type=str, default=DEFAULT_BASE)
    parser.add_argument("--output_dir", type=Path, default=Path("artifacts/adapters"))
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Train each persona on at most 50 total rows for pipeline smoke test.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train_df = pd.read_parquet(args.processed_dir / "train.parquet")
    val_df = pd.read_parquet(args.processed_dir / "val.parquet")

    if args.dry_run:
        train_df = train_df.head(50).copy()

    for persona in tqdm(PERSONAS, desc="Personas"):
        train_persona(
            persona=persona,
            train_df=train_df,
            val_df=val_df,
            base_model=args.base_model,
            output_root=args.output_dir,
            epochs=1 if args.dry_run else args.epochs,
            lr=args.learning_rate,
            max_length=args.max_length,
        )


if __name__ == "__main__":
    main()
