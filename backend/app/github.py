"""Minimal GitHub REST client with a token pool and rate-limit waiting."""

import json
import logging
import os
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import httpx

from .config import settings

log = logging.getLogger(__name__)

API = "https://api.github.com"
# Written to a file, not the DB: with SQLite a second writer would block on the stage's open transaction.
RATE_FILE = settings.data_dir / "github_rate.json"


def read_rate_state() -> dict | None:
    try:
        return json.loads(RATE_FILE.read_text())
    except (OSError, ValueError):
        return None


class GitHubError(Exception):
    pass


@dataclass
class TokenState:
    token: str | None
    remaining: int = 5000
    limit: int = 5000
    reset: int = 0  # epoch seconds


class GitHubClient:
    def __init__(
        self,
        tokens: tuple[str, ...] = settings.github_tokens,
        on_wait: Callable[[int], None] | None = None,
    ):
        """`on_wait(seconds_left)` is called while waiting for a reset; it may raise to abort the wait."""
        self.tokens = [TokenState(t) for t in tokens] or [TokenState(None, remaining=60, limit=60)]
        self.on_wait = on_wait
        self.http = httpx.Client(timeout=60, headers={"Accept": "application/vnd.github+json"})
        self._last_publish = 0.0

    # -- token pool -----------------------------------------------------------------

    def _pick(self) -> TokenState:
        now = time.time()
        for t in self.tokens:
            if t.reset and t.reset <= now:
                t.remaining = t.limit
        usable = [t for t in self.tokens if t.remaining > 1]
        if usable:
            return max(usable, key=lambda t: t.remaining)
        reset_at = min(t.reset for t in self.tokens)
        self._wait_until(reset_at)
        return self._pick()

    def _wait_until(self, reset_at: int):
        self._publish(force=True, waiting_until=reset_at)
        while (left := int(reset_at - time.time()) + 2) > 0:
            log.info("GitHub rate limit exhausted, waiting %ss", left)
            if self.on_wait:
                self.on_wait(left)
            time.sleep(min(left, 5))
        self._publish(force=True)

    def _publish(self, force: bool = False, waiting_until: int | None = None):
        """Store the pool status so the UI can show it."""
        if not force and time.time() - self._last_publish < 5:
            return
        self._last_publish = time.time()
        value = {
            "remaining": sum(t.remaining for t in self.tokens),
            "limit": sum(t.limit for t in self.tokens),
            "reset": min((t.reset for t in self.tokens), default=0),
            "tokens": len([t for t in self.tokens if t.token]),
            "waiting_until": waiting_until,
            "checked_at": int(time.time()),
        }
        try:
            tmp = RATE_FILE.with_suffix(".tmp")
            tmp.write_text(json.dumps(value))
            os.replace(tmp, RATE_FILE)
        except OSError:  # status reporting must never break mining
            log.exception("could not publish rate-limit state")

    # -- requests -------------------------------------------------------------------

    def get(self, path: str, params: dict | None = None) -> httpx.Response:
        url = path if path.startswith("http") else f"{API}{path}"
        for attempt in range(6):
            tok = self._pick()
            headers = {"Authorization": f"Bearer {tok.token}"} if tok.token else {}
            try:
                r = self.http.get(url, params=params, headers=headers)
            except httpx.TransportError as e:
                log.warning("network error %s (attempt %s)", e, attempt)
                time.sleep(2**attempt)
                continue

            if "x-ratelimit-remaining" in r.headers:
                tok.remaining = int(r.headers["x-ratelimit-remaining"])
                tok.limit = int(r.headers.get("x-ratelimit-limit", tok.limit))
                tok.reset = int(r.headers.get("x-ratelimit-reset", tok.reset))
            self._publish()

            if r.status_code in (403, 429) and (tok.remaining == 0 or "retry-after" in r.headers):
                if "retry-after" in r.headers:  # secondary rate limit
                    self._wait_until(int(time.time()) + int(r.headers["retry-after"]))
                else:
                    tok.remaining = 0
                continue
            if r.status_code >= 500:
                time.sleep(2**attempt)
                continue
            if r.status_code >= 400:
                raise GitHubError(f"GET {url} -> {r.status_code}: {r.text[:300]}")
            return r
        raise GitHubError(f"GET {url} failed after retries")

    def paginate(self, path: str, params: dict | None = None, start_page: int = 1) -> Iterator[tuple[int, list]]:
        """Yield (page_number, items) until the last page."""
        page = start_page
        while True:
            r = self.get(path, {**(params or {}), "per_page": 100, "page": page})
            items = r.json()
            if not items:
                return
            yield page, items
            if 'rel="next"' not in r.headers.get("link", ""):
                return
            page += 1

    def close(self):
        self.http.close()
