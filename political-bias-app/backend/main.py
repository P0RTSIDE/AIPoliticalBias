"""PolitiChat Phase 1 — chat, reset, health only."""

from __future__ import annotations

import logging
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from inference import run_chat_turn
from model_loader import load_model_store
from session_store import clear_session, ensure_session, new_session_id

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="PolitiChat", version="1.0.0-phase1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_STORE: dict | None = None
# Slow CPU + use_cache=False often needs >150s per persona; override with env if needed.
PER_PERSONA_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "480"))


def _memory_usage_mb() -> float | None:
    try:
        import psutil  # type: ignore

        rss = psutil.Process(os.getpid()).memory_info().rss
        return round(rss / (1024 * 1024), 1)
    except Exception:
        return None


@app.on_event("startup")
def startup_event() -> None:
    global MODEL_STORE
    logger.info("Loading Phi-3-mini (CPU, float32)…")
    MODEL_STORE = load_model_store(os.environ.get("BASE_MODEL"))
    logger.info("Model ready: %s", MODEL_STORE["base_model"])


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8000)
    session_id: str | None = Field(default=None, max_length=128)


class PersonaReply(BaseModel):
    response: str
    inference_time_seconds: float


class ChatResponse(BaseModel):
    democrat: PersonaReply
    republican: PersonaReply
    centrist: PersonaReply
    session_id: str


class ResetRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=128)


class ResetResponse(BaseModel):
    status: str


@app.get("/health")
def health() -> dict:
    mem = _memory_usage_mb()
    if MODEL_STORE is None:
        return {
            "status": "loading",
            "ready": False,
            "device": None,
            "dtype": None,
            "base_model": None,
            "memory_mb": mem,
        }
    body: dict = {
        "status": "ok",
        "ready": True,
        "device": MODEL_STORE["device"],
        "dtype": MODEL_STORE["dtype"],
        "base_model": MODEL_STORE["base_model"],
        "memory_mb": mem,
    }
    if mem is None:
        body["memory_note"] = "Install psutil for RSS (pip install psutil)"
    return body


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    if MODEL_STORE is None:
        raise HTTPException(status_code=503, detail="Model is still loading.")
    sid = (payload.session_id or "").strip() or new_session_id()
    ensure_session(sid)
    try:
        raw = run_chat_turn(
            MODEL_STORE,
            sid,
            payload.message.strip(),
            PER_PERSONA_TIMEOUT,
        )
    except Exception as exc:
        logger.exception("chat failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return ChatResponse(
        democrat=PersonaReply(
            response=raw["democrat"]["response"],
            inference_time_seconds=float(raw["democrat"]["inference_time_seconds"]),
        ),
        republican=PersonaReply(
            response=raw["republican"]["response"],
            inference_time_seconds=float(raw["republican"]["inference_time_seconds"]),
        ),
        centrist=PersonaReply(
            response=raw["centrist"]["response"],
            inference_time_seconds=float(raw["centrist"]["inference_time_seconds"]),
        ),
        session_id=sid,
    )


@app.post("/reset", response_model=ResetResponse)
def reset(payload: ResetRequest) -> ResetResponse:
    sid = payload.session_id.strip()
    clear_session(sid)  # ok if session did not exist
    return ResetResponse(status="cleared")
