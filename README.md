# Political Bias Personas

A CPU-only application that runs one news text through three LoRA adapters on Phi-3 (Democrat, Republican, Centrist) and shows the three readings together.

Training and inference stay on CPU. There is no cloud LLM API, no CUDA path, and no 4-bit quantization.

The application is in `political-bias-app/`: FastAPI backend, React frontend, and the training scripts. Setup, data format, and the API are documented in `political-bias-app/README.md`.

`Data.py` downloads the Kaggle set `cl0ud0/news-political-bias-classification-dataset` with `kagglehub`. Training expects a CSV with `text` and `label` columns.
