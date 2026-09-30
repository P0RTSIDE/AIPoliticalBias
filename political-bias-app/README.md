# Political bias app (CPU only)

Three LoRA adapters (Democrat, Republican, Centrist) on `microsoft/Phi-3-mini-4k-instruct`, trained from a CSV with `text` and `label` columns. FastAPI serves the adapters. The UI is React and Tailwind.

The stack is CPU only: no CUDA calls, no `bitsandbytes`, and no 4-bit quantization. Inference uses `device_map="cpu"` and bfloat16 when `phi3_compat.cpu_supports_bfloat16()` is true, otherwise float32. `phi3_compat.patch_phi3_rope_config` rewrites Phi-3 `rope_scaling` for Transformers 5.x and the hub remote code, which otherwise raises `KeyError: 'type'`.

## Hardware

Sized for a machine with integrated graphics and 32 GB of RAM. CPU inference is on the order of 30 to 90 seconds per persona.

## Setup

```bash
cd political-bias-app
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\activate`.

Optional environment variables. Defaults are correct when the API is started from `backend/` as below.

- `BASE_MODEL=microsoft/Phi-3-mini-4k-instruct`
- `ADAPTERS_ROOT=../artifacts/adapters`
- `REQUEST_TIMEOUT_SECONDS=120` (per persona, checked inside `generate`)

## Data

```bash
python training/preprocess.py --input_csv data/labeled_news.csv --output_dir artifacts/processed
```

Writes `train.parquet`, `val.parquet`, and `test.parquet` under `artifacts/processed/`.

`Data.py` at the repository root downloads `cl0ud0/news-political-bias-classification-dataset` through `kagglehub`.

## Training

Smoke test: 50 training rows, one epoch per persona.

```bash
python training/train.py --processed_dir artifacts/processed --output_dir artifacts/adapters --dry_run
```

Full run (hours on CPU):

```bash
python training/train.py --processed_dir artifacts/processed --output_dir artifacts/adapters
```

Adapters:

- `artifacts/adapters/democrat/adapter`
- `artifacts/adapters/republican/adapter`
- `artifacts/adapters/centrist/adapter`

Training uses the Hugging Face `Trainer` with `gradient_accumulation_steps=4` and batch size 1.

## Evaluation

```bash
python training/evaluate.py --processed_dir artifacts/processed --adapters_dir artifacts/adapters
```

Writes `artifacts/metrics.json`: accuracy and macro F1 by perplexity routing, plus sample generations.

## API

Start from `political-bias-app/backend` so `ADAPTERS_ROOT=../artifacts/adapters` resolves.

```bash
cd backend
export ADAPTERS_ROOT="../artifacts/adapters"
export BASE_MODEL="microsoft/Phi-3-mini-4k-instruct"
uvicorn main:app --host 0.0.0.0 --port 8000
```

On Windows PowerShell, set variables with `$env:ADAPTERS_ROOT = "..\artifacts\adapters"`.

- `GET /health`: load status, `device: cpu`, base model id, dtype
- `POST /analyze`: body `{ "news": "..." }` (at least 20 characters). Each persona returns `response`, `confidence`, and `inference_seconds`.

Per-persona timeouts use `StoppingCriteria` on each generation step. No background thread shares the `PeftModel`.

## Frontend

```bash
cd frontend
npm install
npm run dev
```

Optional `frontend/.env`: `VITE_API_URL=http://127.0.0.1:8000`.

## Notes

- `pyarrow` is in `requirements.txt` for Parquet.
- Hub `config.json` stores `rope_scaling.rope_type`. Remote `modeling_phi3.py` expects `rope_scaling.type`. The patch runs before `from_pretrained`. A bad download can be cleared by removing the hub cache directory `models--microsoft--Phi-3-mini-4k-instruct` under the Hugging Face cache (`~/.cache/huggingface/hub` on macOS and Linux, `%USERPROFILE%\.cache\huggingface\hub` on Windows).
- Older torch builds have no `torch.cpu.is_bf16_supported`. `phi3_compat.py` falls back to float32.
- `torch==2.3.0` is not published for Python 3.13. `requirements.txt` asks for `torch>=2.6.0`. Python 3.11 matches an exact torch pin when one is required.
