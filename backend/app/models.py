from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from .db import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[int] = mapped_column(primary_key=True)
    owner: Mapped[str] = mapped_column(String(200))
    name: Mapped[str] = mapped_column(String(200))
    url: Mapped[str] = mapped_column(String(500))
    # queued | running | paused | done | failed
    status: Mapped[str] = mapped_column(String(20), default="queued", index=True)
    pause_requested: Mapped[bool] = mapped_column(Boolean, default=False)
    current_stage: Mapped[str | None] = mapped_column(String(50))
    error: Mapped[str | None] = mapped_column(Text)
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime)

    __table_args__ = (UniqueConstraint("owner", "name"),)

    @property
    def full_name(self) -> str:
        return f"{self.owner}/{self.name}"


class StageRun(Base):
    __tablename__ = "stage_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    stage: Mapped[str] = mapped_column(String(50))
    position: Mapped[int] = mapped_column(Integer)
    # pending | running | paused | done | failed
    status: Mapped[str] = mapped_column(String(20), default="pending")
    done: Mapped[int] = mapped_column(Integer, default=0)
    total: Mapped[int | None] = mapped_column(Integer)
    cursor: Mapped[dict] = mapped_column(JSON, default=dict)
    message: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime | None] = mapped_column(DateTime)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime)

    __table_args__ = (UniqueConstraint("project_id", "stage"),)


class Log(Base):
    __tablename__ = "logs"

    id: Mapped[int] = mapped_column(primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    ts: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    level: Mapped[str] = mapped_column(String(10), default="info")
    stage: Mapped[str | None] = mapped_column(String(50))
    message: Mapped[str] = mapped_column(Text)


class Developer(Base):
    __tablename__ = "developers"

    id: Mapped[int] = mapped_column(primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    pseudonym: Mapped[str] = mapped_column(String(40))
    display_name: Mapped[str | None] = mapped_column(String(300))
    login: Mapped[str | None] = mapped_column(String(200))
    is_bot: Mapped[bool] = mapped_column(Boolean, default=False)
    truck_factor_member: Mapped[bool] = mapped_column(Boolean, default=False)
    files_authored: Mapped[int] = mapped_column(Integer, default=0)
    first_activity: Mapped[datetime | None] = mapped_column(DateTime)
    last_activity: Mapped[datetime | None] = mapped_column(DateTime)


class Alias(Base):
    """A raw identity: a git (name, email) pair or a GitHub login."""

    __tablename__ = "aliases"

    id: Mapped[int] = mapped_column(primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    key: Mapped[str] = mapped_column(String(600))  # "git:<email>|<name>" or "gh:<login>"
    kind: Mapped[str] = mapped_column(String(10))  # git | gh
    name: Mapped[str | None] = mapped_column(String(300))
    email: Mapped[str | None] = mapped_column(String(300))
    login: Mapped[str | None] = mapped_column(String(200))
    gh_type: Mapped[str | None] = mapped_column(String(20))  # User | Bot | Organization
    is_bot: Mapped[bool] = mapped_column(Boolean, default=False)
    developer_id: Mapped[int | None] = mapped_column(ForeignKey("developers.id", ondelete="SET NULL"), index=True)

    __table_args__ = (UniqueConstraint("project_id", "key"),)


class Commit(Base):
    __tablename__ = "commits"

    id: Mapped[int] = mapped_column(primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"))
    sha: Mapped[str] = mapped_column(String(40))
    seq: Mapped[int] = mapped_column(Integer)  # chronological order within the project
    alias_id: Mapped[int] = mapped_column(ForeignKey("aliases.id", ondelete="CASCADE"), index=True)
    authored_at: Mapped[datetime] = mapped_column(DateTime)  # UTC
    tz_offset_min: Mapped[int] = mapped_column(Integer)  # author's declared offset
    committed_at: Mapped[datetime] = mapped_column(DateTime)
    is_merge: Mapped[bool] = mapped_column(Boolean, default=False)
    additions: Mapped[int] = mapped_column(Integer, default=0)
    deletions: Mapped[int] = mapped_column(Integer, default=0)
    files_changed: Mapped[int] = mapped_column(Integer, default=0)

    __table_args__ = (
        UniqueConstraint("project_id", "sha"),
        Index("ix_commits_project_seq", "project_id", "seq"),
    )


class CommitFile(Base):
    __tablename__ = "commit_files"

    id: Mapped[int] = mapped_column(primary_key=True)
    commit_id: Mapped[int] = mapped_column(ForeignKey("commits.id", ondelete="CASCADE"), index=True)
    path: Mapped[str] = mapped_column(String(1000))
    old_path: Mapped[str | None] = mapped_column(String(1000))
    change_type: Mapped[str] = mapped_column(String(1))  # A M D R
    additions: Mapped[int] = mapped_column(Integer, default=0)
    deletions: Mapped[int] = mapped_column(Integer, default=0)


class Issue(Base):
    __tablename__ = "issues"

    id: Mapped[int] = mapped_column(primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"))
    number: Mapped[int] = mapped_column(Integer)
    is_pr: Mapped[bool] = mapped_column(Boolean, default=False)
    alias_id: Mapped[int | None] = mapped_column(ForeignKey("aliases.id", ondelete="SET NULL"), index=True)
    title: Mapped[str | None] = mapped_column(Text)
    body: Mapped[str | None] = mapped_column(Text)
    state: Mapped[str] = mapped_column(String(20))
    created_at: Mapped[datetime] = mapped_column(DateTime)
    updated_at: Mapped[datetime] = mapped_column(DateTime)
    closed_at: Mapped[datetime | None] = mapped_column(DateTime)

    __table_args__ = (UniqueConstraint("project_id", "number"),)


class Comment(Base):
    __tablename__ = "comments"

    id: Mapped[int] = mapped_column(primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"))
    gh_id: Mapped[int] = mapped_column(BigInteger)
    kind: Mapped[str] = mapped_column(String(20))  # issue_comment | review_comment | review
    issue_number: Mapped[int] = mapped_column(Integer, index=True)
    alias_id: Mapped[int | None] = mapped_column(ForeignKey("aliases.id", ondelete="SET NULL"), index=True)
    body: Mapped[str | None] = mapped_column(Text)
    review_state: Mapped[str | None] = mapped_column(String(30))
    created_at: Mapped[datetime] = mapped_column(DateTime)
    updated_at: Mapped[datetime | None] = mapped_column(DateTime)

    __table_args__ = (UniqueConstraint("project_id", "kind", "gh_id"),)


class DeveloperMonth(Base):
    __tablename__ = "developer_months"

    id: Mapped[int] = mapped_column(primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"))
    developer_id: Mapped[int] = mapped_column(ForeignKey("developers.id", ondelete="CASCADE"), index=True)
    month: Mapped[str] = mapped_column(String(7))  # YYYY-MM
    commits: Mapped[int] = mapped_column(Integer, default=0)
    additions: Mapped[int] = mapped_column(Integer, default=0)
    deletions: Mapped[int] = mapped_column(Integer, default=0)
    off_hours_commits: Mapped[int] = mapped_column(Integer, default=0)
    weekend_commits: Mapped[int] = mapped_column(Integer, default=0)
    issues_opened: Mapped[int] = mapped_column(Integer, default=0)
    prs_opened: Mapped[int] = mapped_column(Integer, default=0)
    comments: Mapped[int] = mapped_column(Integer, default=0)
    reviews: Mapped[int] = mapped_column(Integer, default=0)
    centrality: Mapped[float] = mapped_column(Float, default=0.0)
    degree: Mapped[int] = mapped_column(Integer, default=0)
    is_core: Mapped[bool] = mapped_column(Boolean, default=False)

    __table_args__ = (UniqueConstraint("project_id", "developer_id", "month"),)


class ProjectMetric(Base):
    __tablename__ = "project_metrics"

    id: Mapped[int] = mapped_column(primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"))
    key: Mapped[str] = mapped_column(String(100))
    value: Mapped[dict] = mapped_column(JSON)

    __table_args__ = (UniqueConstraint("project_id", "key"),)


class AppState(Base):
    __tablename__ = "app_state"

    key: Mapped[str] = mapped_column(String(100), primary_key=True)
    value: Mapped[dict] = mapped_column(JSON)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)
