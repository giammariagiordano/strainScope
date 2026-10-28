"""Emotion classifier chosen in the UI (database) or in the environment (EMOTION_BACKEND, OPENAI_MODEL, OPENAI_API_KEY).

UI values win over the environment; the OpenAI key from the UI is stored in plain text in the local database, like
the GitHub tokens, and is never returned by the API. The emotions stage reads this at every run, so a change applies
to the next project without restarting the worker.
"""

import os
from dataclasses import dataclass

import httpx

from .config import settings
from .db import session_scope
from .models import AppState

KEY = "emotion_classifier"
BACKENDS = ("roberta", "openai")


class InvalidKey(Exception):
    pass


@dataclass(frozen=True)
class EmotionConfig:
    backend: str
    model: str
    api_key: str | None
    key_source: str | None  # "ui", "env" or None


def _stored() -> dict:
    with session_scope() as s:
        state = s.get(AppState, KEY)
        return dict(state.value) if state else {}


def save(**changes):
    """Set (or, with None, clear) backend, model and api_key."""
    with session_scope() as s:
        state = s.get(AppState, KEY)
        value = dict(state.value) if state else {}
        for k, v in changes.items():
            if v is None:
                value.pop(k, None)
            else:
                value[k] = v
        if state:
            state.value = value
        else:
            s.add(AppState(key=KEY, value=value))


def current() -> EmotionConfig:
    ui = _stored()
    env_key = os.environ.get("OPENAI_API_KEY")
    key, source = (ui["api_key"], "ui") if ui.get("api_key") else (env_key, "env") if env_key else (None, None)
    return EmotionConfig(
        backend=ui.get("backend", settings.emotion_backend),
        model=ui.get("model", settings.openai_model),
        api_key=key,
        key_source=source,
    )


def check(key: str, model: str, timeout: float = 15) -> None:
    """Ask OpenAI whether the key is valid and can use the model."""
    try:
        r = httpx.get(f"https://api.openai.com/v1/models/{model}", headers={"Authorization": f"Bearer {key}"}, timeout=timeout)
    except httpx.HTTPError as e:
        raise InvalidKey(f"Could not reach OpenAI: {e}") from e
    if r.status_code == 401:
        raise InvalidKey("OpenAI rejected the key (401).")
    if r.status_code == 404:
        raise InvalidKey(f"The key works but cannot use the model {model!r} (404).")
    if r.status_code >= 400:
        raise InvalidKey(f"OpenAI answered {r.status_code}: {r.text[:200]}")
