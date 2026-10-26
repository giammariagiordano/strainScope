"""Stage framework: context, checkpoints and the ordered stage registry."""

import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from .config import settings
from .github import GitHubClient
from .models import Alias, Log, Project, StageRun, utcnow


class Paused(Exception):
    """Raised at a checkpoint when the user asked to pause."""


class Interrupted(Exception):
    """Raised at a checkpoint when the worker process is shutting down."""


# Set by the worker on SIGINT/SIGTERM; checked at every checkpoint.
STOP = threading.Event()


@dataclass
class ProjectRef:
    id: int
    owner: str
    name: str
    since: datetime | None = None  # see Project.since

    @property
    def since_iso(self) -> str | None:
        return self.since.isoformat() + "Z" if self.since else None

    @property
    def full_name(self) -> str:
        return f"{self.owner}/{self.name}"

    @property
    def repo_path(self) -> Path:
        return settings.repos_dir / f"{self.owner}__{self.name}.git"


class StageContext:
    """What a stage gets to work with.

    `db` is the stage's own session. Data written through it and the cursor are committed
    together by `checkpoint()`, so a crash never leaves data and cursor out of sync.
    """

    def __init__(self, db: Session, project: ProjectRef, run: StageRun):
        self.db = db
        self.project = project
        self.run = run
        self.cursor: dict = dict(run.cursor or {})
        self._github: GitHubClient | None = None
        self._alias_cache: dict[str, int] = {}

    # -- progress / checkpoints -----------------------------------------------------

    def progress(self, done: int, total: int | None = None, message: str | None = None):
        self.run.done = done
        if total is not None:
            self.run.total = total
        if message is not None:
            self.run.message = message

    def checkpoint(self):
        """Persist data + cursor + progress atomically, then honour a pause request."""
        self.run.cursor = dict(self.cursor)
        project = self.db.get(Project, self.project.id)
        project.heartbeat_at = utcnow()
        self.db.commit()
        self.check_pause()

    def check_pause(self):
        if STOP.is_set():
            raise Interrupted()
        self.db.expire_all()
        project = self.db.get(Project, self.project.id)
        if project is None or project.pause_requested:
            raise Paused()

    def log(self, message: str, level: str = "info"):
        self.db.add(Log(project_id=self.project.id, stage=self.run.stage, level=level, message=message))

    # -- helpers --------------------------------------------------------------------

    @property
    def github(self) -> GitHubClient:
        if self._github is None:

            def on_wait(seconds_left: int):
                self.progress(self.run.done, message=f"Waiting for GitHub rate-limit reset ({seconds_left}s)")
                self.checkpoint()

            self._github = GitHubClient(on_wait=on_wait)
        return self._github

    def alias_id(self, key: str, kind: str, **fields) -> int:
        """Get or create an alias by natural key, cached for the stage's lifetime."""
        if key in self._alias_cache:
            return self._alias_cache[key]
        alias = self.db.scalar(select(Alias).where(Alias.project_id == self.project.id, Alias.key == key))
        if alias is None:
            alias = Alias(project_id=self.project.id, key=key, kind=kind, **fields)
            self.db.add(alias)
            self.db.flush()
        elif kind == "gh" and fields.get("gh_type") and not alias.gh_type:
            alias.gh_type = fields["gh_type"]
        self._alias_cache[key] = alias.id
        return alias.id

    def gh_alias_id(self, user: dict | None) -> int | None:
        if not user or not user.get("login"):
            return None
        login = user["login"]
        return self.alias_id(f"gh:{login.lower()}", "gh", login=login, gh_type=user.get("type"))

    def close(self):
        if self._github:
            self._github.close()


class Stage:
    name: str = ""
    label: str = ""
    description: str = ""
    # Exclusive stages load large local models (GPU/CPU memory): the worker runs one of them at a time across
    # the projects it processes in parallel.
    exclusive: bool = False

    def run(self, ctx: StageContext) -> None:  # pragma: no cover - interface
        raise NotImplementedError


def registry() -> list[Stage]:
    from .stages.aggregate import AggregateStage
    from .stages.ai import AISignalsStage
    from .stages.bots import BotsStage
    from .stages.clone import CloneStage
    from .stages.commit_authors import CommitAuthorsStage
    from .stages.commits import CommitsStage
    from .stages.context import PostExitStage, PullOutcomesStage, RepoInfoStage
    from .stages.github_items import CommentsStage, IssuesStage, ReviewsStage
    from .stages.identities import IdentitiesStage
    from .stages.structural import StructuralStage
    from .stages.text import ConversationToxicityStage, EmotionsStage, SentimentStage, ToxicityStage

    return [
        CloneStage(),
        CommitsStage(),
        CommitAuthorsStage(),
        IssuesStage(),
        CommentsStage(),
        ReviewsStage(),
        BotsStage(),
        IdentitiesStage(),
        ToxicityStage(),
        SentimentStage(),
        EmotionsStage(),
        ConversationToxicityStage(),
        AggregateStage(),
        StructuralStage(),
        # Context stages last: adding them to mined projects recomputes nothing else.
        RepoInfoStage(),
        PullOutcomesStage(),
        PostExitStage(),
        AISignalsStage(),
    ]


def stage_names() -> list[str]:
    # A function, not a module constant: computing it at import time makes stage modules
    # (which import this one) circular when they are imported first.
    return [s.name for s in registry()]


def ensure_stage_runs(db: Session, project_id: int) -> bool:
    """Create missing stage rows and keep positions in registry order.

    When a newer version adds a stage in the middle of the pipeline, every later stage is reset to
    `pending` so that it is recomputed with the new data. Returns True if anything was added.
    """
    runs = {r.stage: r for r in db.scalars(select(StageRun).where(StageRun.project_id == project_id))}
    added_at = None
    for pos, stage in enumerate(registry()):
        run = runs.get(stage.name)
        if run is None:
            db.add(StageRun(project_id=project_id, stage=stage.name, position=pos, status="pending", cursor={}))
            added_at = pos if added_at is None else added_at
            continue
        run.position = pos
        if added_at is not None and runs and run.status == "done":
            run.status, run.cursor, run.done, run.total, run.message = "pending", {}, 0, None, None
    return added_at is not None and bool(runs)
