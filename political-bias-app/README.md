# Political bias app (CPU-only)

Three LoRA adapters (Democrat, Republican, Centrist) on **microsoft/Phi-3-mini-4k-instruct**, trained from a Kaggle-style CSV (`text`, `label`), served by **FastAPI** with a **React + Tailwind** UI.

This stack is **CPU-only**: no CUDA calls, no `bitsandbytes`, no 4-bit quantization. Inference uses `device_map="cpu"` and **bfloat16** when the CPU supports it (`phi3_compat.cpu_supports_bfloat16()`), otherwise **float32**. `phi3_compat.patch_phi3_rope_config` fixes Phi-3 `rope_scaling` for Transformers 5.x + hub remote code (`KeyError: 'type'`).

## Hardware

Target: integrated graphics + **32GB RAM**. Expect **~30–90 seconds per persona** per request on CPU.

## Setup

```powershell
cd political-bias-app
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Set env (optional; defaults work when you run from `backend/` as below):

- `BASE_MODEL=microsoft/Phi-3-mini-4k-instruct`
- `ADAPTERS_ROOT=../artifacts/adapters`
- `REQUEST_TIMEOUT_SECONDS=120` (per persona wall-clock, cooperative stop inside `generate`)

## Data pipeline

```powershell
python training\preprocess.py --input_csv path\to\your.csv --output_dir artifacts\processed
```

Writes `train.parquet`, `val.parquet`, `test.parquet` under `artifacts/processed/`.

## Training (CPU; slow)

Smoke test (50 train rows, 1 epoch per persona):

```powershell
python training\train.py --processed_dir artifacts\processed --output_dir artifacts\adapters --dry_run
```

Full run (expect hours on CPU):

```powershell
python training\train.py --processed_dir artifacts\processed --output_dir artifacts\adapters
```

Adapters are saved to:

- `artifacts/adapters/democrat/adapter`
- `artifacts/adapters/republican/adapter`
- `artifacts/adapters/centrist/adapter`

## Evaluation

```powershell
python training\evaluate.py --processed_dir artifacts\processed --adapters_dir artifacts\adapters
```

Writes `artifacts/metrics.json` (accuracy / macro F1 via perplexity routing, sample generations).

## API

From `political-bias-app/backend` (so `ADAPTERS_ROOT=../artifacts/adapters` resolves):

```powershell
cd backend
$env:ADAPTERS_ROOT = "..\artifacts\adapters"
$env:BASE_MODEL = "microsoft/Phi-3-mini-4k-instruct"
uvicorn main:app --host 0.0.0.0 --port 8000
```

- `GET /health` — load status, `device: cpu`, base model id, dtype string
- `POST /analyze` — body `{ "news": "..." }` (min 20 characters). Returns each persona with `response`, `confidence`, and `inference_seconds`.

## Frontend

```powershell
cd frontend
npm install
npm run dev
```

Optional: `frontend/.env` with `VITE_API_URL=http://127.0.0.1:8000`.

## Notes

- Training uses **HuggingFace `Trainer`** with `gradient_accumulation_steps=4` and batch size 1.
- Per-persona timeouts use **`StoppingCriteria`** (checked each generation step); no background threads touching the same `PeftModel`.
- `pyarrow` is listed in `requirements.txt` for Parquet I/O.

## Troubleshooting

- **`KeyError: 'type'` in `modeling_phi3._init_rope`:** Hub `config.json` uses `rope_scaling.rope_type` while remote `modeling_phi3.py` expects `rope_scaling.type`. The repo patches this automatically before `from_pretrained`. If you still see errors after a failed download, clear the model cache and retry:  
  `Remove-Item -Recurse -Force "$env:USERPROFILE\.cache\huggingface\hub\models--microsoft--Phi-3-mini-4k-instruct"` (PowerShell).
- **`torch.cpu` has no `is_bf16_supported`:** Use the updated code path in `phi3_compat.py` (safe fallback to float32).
- **`torch==2.3.0` not found on Python 3.13:** Use `torch>=2.6.0` (see `requirements.txt`) or use **Python 3.11** if you need an exact torch pin.
