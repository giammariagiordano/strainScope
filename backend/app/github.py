"""Minimal GitHub REST client with a token pool and rate-limit waiting."""

import json
import logging
import os
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import httpx

from .config import settings
from .tokens import all_tokens

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
    REFRESH_S = 30

    def __init__(
        self,
        tokens: tuple[str, ...] | None = None,
        on_wait: Callable[[int], None] | None = None,
    ):
        """`on_wait(seconds_left)` is called while waiting for a reset; it may raise to abort the wait.

        Without explicit `tokens`, the pool is loaded from the environment and the database and
        refreshed every REFRESH_S seconds, so tokens added from the UI are used without a restart.
        """
        self._fixed = tokens is not None
        self.tokens: list[TokenState] = []
        self._sync(list(tokens) if self._fixed else all_tokens())
        self.on_wait = on_wait
        # Renamed repositories answer 301 with a JSON body: follow the redirect instead of reading it as data.
        self.http = httpx.Client(timeout=60, follow_redirects=True, headers={"Accept": "application/vnd.github+json"})
        self._last_publish = 0.0
        self._last_refresh = time.time()
        self.suspicious_pages = 0

    # -- token pool -----------------------------------------------------------------

    def _sync(self, tokens: list[str]):
        """Replace the pool, keeping the quota state of tokens that are still present."""
        known = {t.token: t for t in self.tokens}
        self.tokens = [known.get(tok) or TokenState(tok) for tok in tokens] or [
            known.get(None) or TokenState(None, remaining=60, limit=60)
        ]

    def _refresh(self):
        if self._fixed or time.time() - self._last_refresh < self.REFRESH_S:
            return
        self._last_refresh = time.time()
        try:
            self._sync(all_tokens())
        except Exception:  # a DB hiccup must not break mining
            log.exception("could not reload GitHub tokens")

    def _pick(self) -> TokenState:
        self._refresh()
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
            # A token added from the UI ends the wait early.
            before = {t.token for t in self.tokens}
            self._refresh()
            if any(t.token not in before for t in self.tokens):
                break
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

    def get_optional(self, path: str, params: dict | None = None) -> dict | None:
        """GET a single resource; None if GitHub says it does not exist (404) or was deleted (410)."""
        try:
            return self.get(path, params).json()
        except GitHubError as e:
            if " -> 404" in str(e) or " -> 410" in str(e):
                return None
            raise

    EMPTY_RETRIES = (2, 4, 8, 16)  # seconds

    def paginate(
        self, path: str, params: dict | None = None, start_page: int = 1, retry_empty: bool = True
    ) -> Iterator[tuple[int, list]]:
        """Yield (page_number, items) until the last page, following `Link: rel="next"`.

        GitHub sometimes answers 200 with an empty or truncated page that should hold data (seen on the
        issues and comments endpoints). A page is suspicious when it is empty although it is the first page
        or the previous page was full, or when it is short but still has a next link; such pages are fetched
        again with backoff before being accepted. Disable `retry_empty` for endpoints where an empty first
        page is common and legitimate (e.g. the reviews of a pull request).
        """
        page = start_page
        url, query = path, {**(params or {}), "per_page": 100, "page": page}
        prev_full = False
        while True:
            # An empty first page is often legitimate (no issues yet): retry briefly. After a full page, an
            # empty or short-with-next page is almost certainly wrong: retry with the full backoff.
            ladder = self.EMPTY_RETRIES if prev_full else self.EMPTY_RETRIES[:2]
            for attempt, wait in enumerate((0, *ladder)):
                if wait:
                    self.suspicious_pages += 1
                    log.warning("GitHub returned a suspicious page for %s (page %s); retry %s in %ss", path, page, attempt, wait)
                    if self.on_wait:
                        self.on_wait(wait)
                    time.sleep(wait)
                r = self.get(url, query)
                items = r.json()
                if not isinstance(items, list):
                    raise GitHubError(f"GET {path}: expected a list, got {str(items)[:200]}")
                nxt = r.links.get("next", {}).get("url")
                suspicious = (not items and (page == start_page or prev_full)) or (0 < len(items) < 100 and nxt)
                if not (retry_empty and suspicious):
                    break
            if not items:
                return
            yield page, items
            if not nxt:
                return
            prev_full = len(items) == 100
            url, query = nxt, None  # the next URL already carries every parameter, including the cursor
            page += 1

    def close(self):
        self.http.close()
