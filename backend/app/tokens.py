"""GitHub tokens from the environment (GITHUB_TOKENS) and from the UI (database)."""

import httpx
from sqlalchemy import select

from .config import settings
from .db import session_scope
from .models import GitHubToken


class InvalidToken(Exception):
    pass


def db_tokens() -> list[str]:
    with session_scope() as s:
        return list(s.scalars(select(GitHubToken.token).where(GitHubToken.valid).order_by(GitHubToken.id)))


def all_tokens() -> list[str]:
    """Environment tokens first, then valid UI tokens, without duplicates."""
    out: list[str] = []
    for t in [*settings.github_tokens, *db_tokens()]:
        if t not in out:
            out.append(t)
    return out


def check(token: str, timeout: float = 15) -> dict:
    """Ask GitHub who owns the token and how much quota is left."""
    try:
        r = httpx.get(
            "https://api.github.com/user",
            headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"},
            timeout=timeout,
        )
    except httpx.HTTPError as e:
        raise InvalidToken(f"Could not reach GitHub: {e}") from e
    if r.status_code == 401:
        raise InvalidToken("GitHub rejected the token (401 Bad credentials).")
    if r.status_code >= 400:
        raise InvalidToken(f"GitHub answered {r.status_code}: {r.text[:200]}")
    return {
        "login": r.json().get("login"),
        "remaining": int(r.headers.get("x-ratelimit-remaining", 0)),
        "limit": int(r.headers.get("x-ratelimit-limit", 0)),
        "reset": int(r.headers.get("x-ratelimit-reset", 0)),
    }
