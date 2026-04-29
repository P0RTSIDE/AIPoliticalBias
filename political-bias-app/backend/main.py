import logging
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from inference import run_all_personas_sequential
from model_loader import load_models_dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnalyzeRequest(BaseModel):
    news: str = Field(min_length=20, max_length=15000)


class PersonaPayload(BaseModel):
    response: str
    confidence: float
    inference_seconds: float | None = None


class AnalyzeResponse(BaseModel):
    democrat: PersonaPayload
    republican: PersonaPayload
    centrist: PersonaPayload


app = FastAPI(title="Political Bias Persona API (CPU)", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_STORE: dict | None = None
PER_PERSONA_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "120"))


@app.on_event("startup")
def startup_event() -> None:
    global MODEL_STORE
    adapters = os.getenv("ADAPTERS_ROOT", "../artifacts/adapters")
    adapters_root = Path(adapters).resolve()
    base_model = os.getenv("BASE_MODEL")
    logger.info("Loading models from %s (base=%s)", adapters_root, base_model or "default")
    MODEL_STORE = load_models_dict(adapters_root=adapters_root, base_model=base_model)
    logger.info("Models ready on device=%s dtype=%s", MODEL_STORE["device"], MODEL_STORE["dtype"])


@app.get("/health")
def health() -> dict:
    if MODEL_STORE is None:
        return {
            "status": "loading",
            "ready": False,
            "device": None,
            "base_model": None,
            "dtype": None,
        }
    return {
        "status": "ok",
        "ready": True,
        "device": MODEL_STORE["device"],
        "base_model": MODEL_STORE["base_model"],
        "dtype": MODEL_STORE["dtype"],
    }


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(payload: AnalyzeRequest) -> AnalyzeResponse:
    if MODEL_STORE is None:
        raise HTTPException(status_code=503, detail="Models are still loading.")
    try:
        raw = run_all_personas_sequential(
            MODEL_STORE,
            payload.news.strip(),
            PER_PERSONA_TIMEOUT,
        )
    except Exception as exc:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    def pack(key: str) -> PersonaPayload:
        r = raw[key]
        return PersonaPayload(
            response=r["response"],
            confidence=float(r["confidence"]),
            inference_seconds=float(r["inference_seconds"])
            if r.get("inference_seconds") is not None
            else None,
        )

    return AnalyzeResponse(
        democrat=pack("democrat"),
        republican=pack("republican"),
        centrist=pack("centrist"),
    )
