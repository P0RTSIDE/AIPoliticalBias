import argparse
import html
import json
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample


VALID_LABELS = {"democrat", "republican", "centrist"}

LABEL_ALIASES: dict[str, str] = {
    "democrat": "democrat",
    "democratic": "democrat",
    "liberal": "democrat",
    "left": "democrat",
    "left-leaning": "democrat",
    "left leaning": "democrat",
    "republican": "republican",
    "conservative": "republican",
    "right": "republican",
    "right-leaning": "republican",
    "right leaning": "republican",
    "gop": "republican",
    "centrist": "centrist",
    "center": "centrist",
    "centre": "centrist",
    "moderate": "centrist",
    "neutral": "centrist",
    "independent": "centrist",
    "middle": "centrist",
}


def normalize_text(text: str) -> str:
    text = html.unescape(str(text))
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def map_label(raw: str) -> str | None:
    key = raw.strip().lower()
    if key in VALID_LABELS:
        return key
    return LABEL_ALIASES.get(key)


def stratified_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_parts = []
    val_parts = []
    test_parts = []
    for label in sorted(df["label"].unique()):
        class_df = df[df["label"] == label]
        train_df, temp_df = train_test_split(
            class_df, test_size=0.2, random_state=42, shuffle=True
        )
        val_df, test_df = train_test_split(
            temp_df, test_size=0.5, random_state=42, shuffle=True
        )
        train_parts.append(train_df)
        val_parts.append(val_df)
        test_parts.append(test_df)
    train = pd.concat(train_parts).sample(frac=1, random_state=42).reset_index(drop=True)
    val = pd.concat(val_parts).sample(frac=1, random_state=42).reset_index(drop=True)
    test = pd.concat(test_parts).sample(frac=1, random_state=42).reset_index(drop=True)
    return train, val, test


def oversample_train(train_df: pd.DataFrame) -> pd.DataFrame:
    counts = train_df["label"].value_counts()
    target = counts.max()
    balanced_parts = []
    for label in counts.index:
        class_df = train_df[train_df["label"] == label]
        if len(class_df) < target:
            class_df = resample(
                class_df,
                replace=True,
                n_samples=target,
                random_state=42,
            )
        balanced_parts.append(class_df)
    return pd.concat(balanced_parts).sample(frac=1, random_state=42).reset_index(drop=True)


def preprocess(input_csv: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_csv)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("Expected CSV columns: text, label")

    df = df[["text", "label"]].copy()
    df["text"] = df["text"].astype(str).map(normalize_text)
    df["label"] = df["label"].astype(str).str.strip().str.lower().map(map_label)
    df = df.dropna(subset=["text", "label"])
    df = df[df["text"].str.len() >= 50]
    df = df[df["label"].isin(VALID_LABELS)].reset_index(drop=True)

    train_df, val_df, test_df = stratified_split(df)
    train_balanced_df = oversample_train(train_df)

    train_balanced_df.to_parquet(output_dir / "train.parquet", index=False)
    val_df.to_parquet(output_dir / "val.parquet", index=False)
    test_df.to_parquet(output_dir / "test.parquet", index=False)

    summary = {
        "raw_rows": int(len(df)),
        "train_rows_balanced": int(len(train_balanced_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "label_distribution_train": train_balanced_df["label"].value_counts().to_dict(),
        "label_distribution_val": val_df["label"].value_counts().to_dict(),
        "label_distribution_test": test_df["label"].value_counts().to_dict(),
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("artifacts/processed"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    preprocess(args.input_csv, args.output_dir)
