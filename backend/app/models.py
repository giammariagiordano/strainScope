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
    # Mine GitHub API data (issues, comments, reviews, PR outcomes, commit-author links) only from this date;
    # git history is always complete (it is local and cheap). None = everything.
    since: Mapped[datetime | None] = mapped_column(DateTime)

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
    # Public GitHub activity elsewhere in the 12 months after the last month active here (post_exit stage).
    # after_exit_from is set once checked; the counts stay None if the account no longer exists.
    after_exit_from: Mapped[datetime | None] = mapped_column(DateTime)
    after_exit_contributions: Mapped[int | None] = mapped_column(Integer)  # other public repositories
    after_exit_repos: Mapped[int | None] = mapped_column(Integer)
    after_exit_private: Mapped[int | None] = mapped_column(Integer)  # restricted (private) contributions


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
    # AI agent account (ai_signals stage): tool name and role, "author" (opens PRs, commits) or "reviewer".
    ai_tool: Mapped[str | None] = mapped_column(String(30))
    ai_role: Mapped[str | None] = mapped_column(String(10))

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
    # AI tool credited by the commit (co-author trailer, "Generated with" line, or agent author); ai_signals stage.
    ai_tool: Mapped[str | None] = mapped_column(String(30))

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
    comments_count: Mapped[int | None] = mapped_column(Integer)  # as reported by GitHub, to verify completeness
    # Text classification of title + body: 1 / 0, or -1 when there is nothing to classify (bot, empty).
    toxic: Mapped[int | None] = mapped_column(Integer)
    toxicity_p: Mapped[float | None] = mapped_column(Float)
    sentiment: Mapped[str | None] = mapped_column(String(10))  # positive | neutral | negative | none
    # EmoTxt emotions of title + body: comma-separated (anger, joy, love, sadness, fear, surprise), "" = none,
    # "-" = nothing to classify; None = not classified.
    emotions: Mapped[str | None] = mapped_column(String(60))
    # Whole-conversation toxicity (body + conversation comments): 1 / 0, -1 = nothing to classify. docs/TOXICITY.md
    conv_toxic: Mapped[int | None] = mapped_column(Integer)
    conv_toxicity_score: Mapped[float | None] = mapped_column(Float)
    # Pull-request outcome (pr_outcomes stage); None = not fetched.
    merged: Mapped[bool | None] = mapped_column(Boolean)
    merged_at: Mapped[datetime | None] = mapped_column(DateTime)
    merged_by_alias_id: Mapped[int | None] = mapped_column(ForeignKey("aliases.id", ondelete="SET NULL"))
    # AI involvement of a PR (ai_signals stage): source "agent" (opened by an agent account) or "disclosed"
    # (opened by a person whose description credits an AI tool); None = no declared AI.
    ai_tool: Mapped[str | None] = mapped_column(String(30))
    ai_source: Mapped[str | None] = mapped_column(String(10))
    # GitHub reactions on the issue/PR description (app.psych.reactions); None = not mined.
    reactions_pos: Mapped[int | None] = mapped_column(Integer)
    reactions_neg: Mapped[int | None] = mapped_column(Integer)

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
    toxic: Mapped[int | None] = mapped_column(Integer)  # 1 / 0, -1 = nothing to classify
    toxicity_p: Mapped[float | None] = mapped_column(Float)
    sentiment: Mapped[str | None] = mapped_column(String(10))
    emotions: Mapped[str | None] = mapped_column(String(60))  # as Issue.emotions
    updated_at: Mapped[datetime | None] = mapped_column(DateTime)
    # GitHub reactions on the comment (app.psych.reactions); None = not mined (and for reviews, not available).
    reactions_pos: Mapped[int | None] = mapped_column(Integer)
    reactions_neg: Mapped[int | None] = mapped_column(Integer)

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
    messages: Mapped[int | None] = mapped_column(Integer)  # classified issue bodies, comments, reviews
    toxic_emitted: Mapped[int | None] = mapped_column(Integer)
    received: Mapped[int | None] = mapped_column(Integer)  # classified messages by others on own issues/PRs
    toxic_received: Mapped[int | None] = mapped_column(Integer)
    negative_emitted: Mapped[int | None] = mapped_column(Integer)
    negative_received: Mapped[int | None] = mapped_column(Integer)
    threads: Mapped[int | None] = mapped_column(Integer)  # conversations the developer wrote in this month
    toxic_threads: Mapped[int | None] = mapped_column(Integer)  # ... of which classified as toxic conversations
    # Psycholinguistic markers (app.psych) of the developer's own messages ...
    psych_messages: Mapped[int | None] = mapped_column(Integer)  # human messages with at least one word
    words: Mapped[int | None] = mapped_column(Integer)
    we_words: Mapped[int | None] = mapped_column(Integer)
    i_words: Mapped[int | None] = mapped_column(Integer)
    thanks_emitted: Mapped[int | None] = mapped_column(Integer)  # messages expressing gratitude
    # ... and of what others (humans) write on the developer's issues/PRs, and react to the developer's items.
    replies_received: Mapped[int | None] = mapped_column(Integer)
    thanks_received: Mapped[int | None] = mapped_column(Integer)
    reacted_items: Mapped[int | None] = mapped_column(Integer)  # own issues/PRs/comments with reaction data
    reactions_pos_received: Mapped[int | None] = mapped_column(Integer)
    reactions_neg_received: Mapped[int | None] = mapped_column(Integer)
    # Emotions (EmoTxt) of the developer's messages, and anger in what others write on the developer's issues/PRs.
    emotion_messages: Mapped[int | None] = mapped_column(Integer)  # classified messages
    anger_emitted: Mapped[int | None] = mapped_column(Integer)
    joy_emitted: Mapped[int | None] = mapped_column(Integer)
    love_emitted: Mapped[int | None] = mapped_column(Integer)
    sadness_emitted: Mapped[int | None] = mapped_column(Integer)
    fear_emitted: Mapped[int | None] = mapped_column(Integer)
    surprise_emitted: Mapped[int | None] = mapped_column(Integer)
    emotion_received: Mapped[int | None] = mapped_column(Integer)
    anger_received: Mapped[int | None] = mapped_column(Integer)
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


class GitHubToken(Base):
    """Token added from the UI. Stored in plain text in the local database; never returned by the API."""

    __tablename__ = "github_tokens"

    id: Mapped[int] = mapped_column(primary_key=True)
    token: Mapped[str] = mapped_column(Text, unique=True)
    login: Mapped[str | None] = mapped_column(String(200))
    last4: Mapped[str] = mapped_column(String(4))
    valid: Mapped[bool] = mapped_column(Boolean, default=True)
    remaining: Mapped[int | None] = mapped_column(Integer)
    limit: Mapped[int | None] = mapped_column(Integer)
    reset: Mapped[int | None] = mapped_column(Integer)
    added_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    checked_at: Mapped[datetime | None] = mapped_column(DateTime)
