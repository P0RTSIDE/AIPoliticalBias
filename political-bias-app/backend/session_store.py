"""In-memory per-session, per-persona chat history (Phase 1)."""

from __future__ import annotations

import secrets
from typing import Any

PERSONAS = ("democrat", "republican", "centrist")

# session_id -> { persona -> list of {"role": "user"|"assistant", "content": str} }
_sessions: dict[str, dict[str, list[dict[str, str]]]] = {}


def new_session_id() -> str:
    return secrets.token_urlsafe(16)


def ensure_session(session_id: str) -> None:
    if session_id not in _sessions:
        _sessions[session_id] = {p: [] for p in PERSONAS}


def get_history(session_id: str, persona: str) -> list[dict[str, str]]:
    ensure_session(session_id)
    return _sessions[session_id][persona]


def append_turn(session_id: str, persona: str, user_text: str, assistant_text: str) -> None:
    ensure_session(session_id)
    h = _sessions[session_id][persona]
    h.append({"role": "user", "content": user_text})
    h.append({"role": "assistant", "content": assistant_text})


def clear_session(session_id: str) -> bool:
    if session_id in _sessions:
        del _sessions[session_id]
        return True
    return False


def snapshot(session_id: str) -> dict[str, Any]:
    ensure_session(session_id)
    return {p: list(_sessions[session_id][p]) for p in PERSONAS}
