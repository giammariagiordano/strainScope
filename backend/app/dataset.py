"""Analysis-ready tables built from the mined data (bots excluded, pseudonyms only)."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from .config import settings
from .models import Developer, DeveloperMonth, Issue, Project

LEVELS = ("developer", "developer_month")


@dataclass(frozen=True)
class Variable:
    key: str
    label: str
    description: str
    kind: str = "count"  # count | share | score | binary


DEVELOPER_MONTH_VARS = [
    Variable("commits", "Commits", "Commits authored in the month."),
    Variable("churn", "Churn", "Lines added + deleted in the month."),
    Variable("off_hours_share", "Off-hours share", "Share of the month's commits on weekends or outside working hours (author's local time). Undefined with no commits.", "share"),
    Variable("weekend_share", "Weekend share", "Share of the month's commits made on Saturday or Sunday. Undefined with no commits.", "share"),
    Variable("issues_opened", "Issues opened", "Issues opened in the month."),
    Variable("prs_opened", "PRs opened", "Pull requests opened in the month."),
    Variable("comments", "Comments", "Issue/PR comments and inline review comments written in the month."),
    Variable("reviews", "Reviews", "Pull-request reviews submitted in the month."),
    Variable("centrality", "Centrality", "Eigenvector centrality in the developer network (sliding window).", "score"),
    Variable("degree", "Degree", "Number of distinct collaborators in the developer network (sliding window)."),
    Variable("is_core", "Core", "1 if the developer is core in the month (cumulative-centrality threshold).", "binary"),
    Variable("messages", "Messages", "Classified messages written (issue/PR bodies, comments, reviews)."),
    Variable("toxic_emitted_share", "Toxicity emitted", "Share of the month's messages classified as toxic by ToxiCR.", "share"),
    Variable("toxic_received_share", "Toxicity received", "Share of messages written by others on the developer's issues/PRs that ToxiCR classifies as toxic.", "share"),
    Variable("negative_emitted_share", "Negativity emitted", "Share of the month's messages with negative polarity (Senti4SD).", "share"),
    Variable("negative_received_share", "Negativity received", "Share of messages by others on the developer's issues/PRs with negative polarity (Senti4SD).", "share"),
    Variable("toxic_thread_share", "Toxic conversations joined", "Share of the conversations the developer wrote in this month that are toxic (conversation classifier, docs/TOXICITY.md).", "share"),
    Variable("community_toxicity", "Community toxicity", "Share of the project's conversations opened this month that are toxic (same for every developer of the project).", "share"),
    Variable("pause_start", "Long pause starts", "1 if this month is followed by at least the threshold of inactivity and the developer later returns.", "binary"),
    Variable("exit", "Exit month", "1 if this is the developer's last active month and they then stay inactive for at least the abandonment threshold before the end of observation.", "binary"),
]  # fmt: skip

DEVELOPER_VARS = [
    Variable("active_months", "Active months", "Months with any recorded activity."),
    Variable("tenure_months", "Tenure (months)", "Months between first and last recorded activity, inclusive."),
    Variable("core_share", "Core share", "Share of active months classified as core.", "share"),
    Variable("truck_factor_member", "Truck-factor member", "1 if the developer is in the project's truck-factor set.", "binary"),
    Variable("files_authored", "Files authored", "Files at HEAD where the developer is an author by DOA."),
    Variable("commits", "Commits", "Total commits."),
    Variable("commits_per_month", "Commits / active month", "Total commits divided by active months.", "score"),
    Variable("churn", "Churn", "Total lines added + deleted."),
    Variable("off_hours_share", "Off-hours share", "Share of all commits on weekends or outside working hours. Undefined with no commits.", "share"),
    Variable("weekend_share", "Weekend share", "Share of all commits on Saturday or Sunday. Undefined with no commits.", "share"),
    Variable("issues_prs_opened", "Issues + PRs opened", "Total issues and pull requests opened."),
    Variable("comments", "Comments", "Total issue/PR and review comments."),
    Variable("reviews", "Reviews", "Total pull-request reviews submitted."),
    Variable("mean_centrality", "Mean centrality", "Mean monthly eigenvector centrality over active months.", "score"),
    Variable("max_centrality", "Max centrality", "Highest monthly eigenvector centrality.", "score"),
    Variable("inactive_months_at_end", "Inactive months at end", "Months between the developer's last activity and the project's last observed month."),
    Variable("messages", "Messages", "Classified messages written over all active months."),
    Variable("toxic_emitted_share", "Toxicity emitted", "Share of all messages classified as toxic by ToxiCR.", "share"),
    Variable("toxic_received_share", "Toxicity received", "Share of messages by others on the developer's issues/PRs classified as toxic.", "share"),
    Variable("negative_emitted_share", "Negativity emitted", "Share of all messages with negative polarity (Senti4SD).", "share"),
    Variable("negative_received_share", "Negativity received", "Share of messages by others on the developer's issues/PRs with negative polarity.", "share"),
    Variable("toxic_thread_share", "Toxic conversations joined", "Share of all conversations the developer wrote in that are toxic.", "share"),
    Variable("community_toxicity", "Community toxicity", "Share of all the project's conversations that are toxic (project-level).", "share"),
    Variable("long_pauses", "Long pauses", "Pauses of at least the threshold after which the developer came back."),
    Variable("abandoned", "Abandoned", "1 if the developer has been inactive for at least the abandonment threshold at the end of observation. Right-censored: recent leavers count as 0.", "binary"),
]  # fmt: skip

VARIABLES = {"developer": DEVELOPER_VARS, "developer_month": DEVELOPER_MONTH_VARS}


def _month_diff(a: str, b: str) -> int:
    ya, ma = map(int, a.split("-"))
    yb, mb = map(int, b.split("-"))
    return (yb * 12 + mb) - (ya * 12 + ma)


def community_toxicity(db: Session, project_ids: list[int], by_month: bool = True) -> dict:
    """Share of classified conversations that are toxic, per (project, month opened) or per project.
    Projects without classified conversations are absent (not measured)."""
    rows = db.execute(
        select(Issue.project_id, Issue.created_at, Issue.conv_toxic)
        .where(Issue.project_id.in_(project_ids), Issue.conv_toxic.in_([0, 1]))
    ).all()
    if not rows:
        return {}
    d = pd.DataFrame(rows, columns=["project_id", "created_at", "toxic"])
    if by_month:
        d["month"] = d["created_at"].map(lambda t: f"{t.year:04d}-{t.month:02d}")
        return d.groupby(["project_id", "month"])["toxic"].mean().to_dict()
    return d.groupby("project_id")["toxic"].mean().to_dict()


def developer_month_frame(
    db: Session, project_ids: list[int] | None = None, abandon_months: int | None = None
) -> pd.DataFrame:
    """One row per active (human) developer × project × month."""
    abandon_months = abandon_months or settings.abandon_months
    q = (
        select(
            Project.owner, Project.name, Developer.pseudonym, Developer.truck_factor_member,
            DeveloperMonth.project_id, DeveloperMonth.developer_id, DeveloperMonth.month,
            DeveloperMonth.commits, DeveloperMonth.additions, DeveloperMonth.deletions,
            DeveloperMonth.off_hours_commits, DeveloperMonth.weekend_commits,
            DeveloperMonth.issues_opened, DeveloperMonth.prs_opened, DeveloperMonth.comments,
            DeveloperMonth.reviews, DeveloperMonth.centrality, DeveloperMonth.degree, DeveloperMonth.is_core,
            DeveloperMonth.messages, DeveloperMonth.toxic_emitted, DeveloperMonth.received,
            DeveloperMonth.toxic_received, DeveloperMonth.negative_emitted, DeveloperMonth.negative_received,
            DeveloperMonth.threads, DeveloperMonth.toxic_threads,
        )
        .join(Developer, Developer.id == DeveloperMonth.developer_id)
        .join(Project, Project.id == DeveloperMonth.project_id)
        .where(~Developer.is_bot)
        .order_by(DeveloperMonth.project_id, DeveloperMonth.developer_id, DeveloperMonth.month)
    )  # fmt: skip
    if project_ids:
        q = q.where(DeveloperMonth.project_id.in_(project_ids))
    df = pd.DataFrame(db.execute(q).mappings().all())
    if df.empty:
        return pd.DataFrame(columns=["project", "developer", "month", *(v.key for v in DEVELOPER_MONTH_VARS)])
    df["project"] = df["owner"] + "/" + df["name"]
    df["developer"] = df["pseudonym"]
    df["churn"] = df["additions"] + df["deletions"]
    commits = df["commits"].replace(0, np.nan)
    df["off_hours_share"] = df["off_hours_commits"] / commits
    df["weekend_share"] = df["weekend_commits"] / commits
    df["is_core"] = df["is_core"].astype(int)
    strain = ["messages", "toxic_emitted", "received", "toxic_received", "negative_emitted", "negative_received",
              "threads", "toxic_threads"]  # fmt: skip
    df[strain] = df[strain].astype(float)  # None (not measured) -> NaN
    df["toxic_emitted_share"] = df["toxic_emitted"] / df["messages"].replace(0, np.nan)
    df["toxic_received_share"] = df["toxic_received"] / df["received"].replace(0, np.nan)
    df["negative_emitted_share"] = df["negative_emitted"] / df["messages"].replace(0, np.nan)
    df["negative_received_share"] = df["negative_received"] / df["received"].replace(0, np.nan)
    df["toxic_thread_share"] = df["toxic_threads"] / df["threads"].replace(0, np.nan)
    climate = community_toxicity(db, df["project_id"].unique().tolist())
    df["community_toxicity"] = [climate.get((p, m), np.nan) for p, m in zip(df["project_id"], df["month"])]
    df["truck_factor_member"] = df["truck_factor_member"].astype(int)
    # Abandonment: inactivity gap between a developer's last month and the project's last observed month.
    project_end = df.groupby("project_id")["month"].transform("max")
    last = df.groupby(["project_id", "developer_id"])["month"].transform("max")
    gap = pd.Series([_month_diff(a, b) for a, b in zip(last, project_end)], index=df.index)
    df["inactive_months_at_end"] = gap
    df["exit"] = ((df["month"] == last) & (gap >= abandon_months)).astype(int)
    # Long pause with return: gap to the developer's next active month.
    midx = df["month"].map(lambda m: _month_diff("0000-01", m))
    nxt = midx.groupby([df["project_id"], df["developer_id"]]).shift(-1)
    df["pause_start"] = ((nxt - midx - 1) >= abandon_months).astype(int)
    return df[
        ["project_id", "developer_id", "project", "developer", "month", "truck_factor_member",
         "off_hours_commits", "weekend_commits", "inactive_months_at_end", "toxic_emitted", "received",
         "toxic_received", "negative_emitted", "negative_received", "threads", "toxic_threads",
         *(v.key for v in DEVELOPER_MONTH_VARS)]
    ]  # fmt: skip


def developer_frame(
    db: Session, project_ids: list[int] | None = None, abandon_months: int | None = None
) -> pd.DataFrame:
    """One row per (human) developer × project, aggregated over all active months."""
    abandon_months = abandon_months or settings.abandon_months
    dm = developer_month_frame(db, project_ids, abandon_months)
    if dm.empty:
        return pd.DataFrame(columns=["project", "developer", *(v.key for v in DEVELOPER_VARS)])
    files = dict(
        db.execute(select(Developer.id, Developer.files_authored).where(Developer.id.in_(dm["developer_id"].unique().tolist())))
        .tuples()
        .all()
    )
    g = dm.groupby(["project_id", "developer_id", "project", "developer"], sort=False)
    out = g.agg(
        first_month=("month", "min"),
        last_month=("month", "max"),
        active_months=("month", "count"),
        core_months=("is_core", "sum"),
        truck_factor_member=("truck_factor_member", "max"),
        commits=("commits", "sum"),
        churn=("churn", "sum"),
        off_hours_commits=("off_hours_commits", "sum"),
        weekend_commits=("weekend_commits", "sum"),
        issues_opened=("issues_opened", "sum"),
        prs_opened=("prs_opened", "sum"),
        comments=("comments", "sum"),
        reviews=("reviews", "sum"),
        mean_centrality=("centrality", "mean"),
        max_centrality=("centrality", "max"),
        inactive_months_at_end=("inactive_months_at_end", "first"),
        abandoned=("exit", "max"),
        long_pauses=("pause_start", "sum"),
        messages=("messages", lambda x: x.sum(min_count=1)),
        toxic_emitted=("toxic_emitted", lambda x: x.sum(min_count=1)),
        received=("received", lambda x: x.sum(min_count=1)),
        toxic_received=("toxic_received", lambda x: x.sum(min_count=1)),
        negative_emitted=("negative_emitted", lambda x: x.sum(min_count=1)),
        negative_received=("negative_received", lambda x: x.sum(min_count=1)),
        threads=("threads", lambda x: x.sum(min_count=1)),
        toxic_threads=("toxic_threads", lambda x: x.sum(min_count=1)),
    ).reset_index()
    out["tenure_months"] = [_month_diff(a, b) + 1 for a, b in zip(out["first_month"], out["last_month"])]
    out["core_share"] = out["core_months"] / out["active_months"]
    out["files_authored"] = out["developer_id"].map(files).fillna(0).astype(int)
    out["commits_per_month"] = out["commits"] / out["active_months"]
    commits = out["commits"].replace(0, np.nan)
    out["off_hours_share"] = out["off_hours_commits"] / commits
    out["weekend_share"] = out["weekend_commits"] / commits
    out["issues_prs_opened"] = out["issues_opened"] + out["prs_opened"]
    out["toxic_emitted_share"] = out["toxic_emitted"] / out["messages"].replace(0, np.nan)
    out["toxic_received_share"] = out["toxic_received"] / out["received"].replace(0, np.nan)
    out["negative_emitted_share"] = out["negative_emitted"] / out["messages"].replace(0, np.nan)
    out["negative_received_share"] = out["negative_received"] / out["received"].replace(0, np.nan)
    out["toxic_thread_share"] = out["toxic_threads"] / out["threads"].replace(0, np.nan)
    overall = community_toxicity(db, out["project_id"].unique().tolist(), by_month=False)
    out["community_toxicity"] = out["project_id"].map(overall)
    return out[
        ["project_id", "developer_id", "project", "developer", "first_month", "last_month",
         *(v.key for v in DEVELOPER_VARS)]
    ]  # fmt: skip


def frame(
    db: Session, level: str, project_ids: list[int] | None = None, abandon_months: int | None = None
) -> pd.DataFrame:
    if level not in LEVELS:
        raise ValueError(f"unknown level {level}")
    build = developer_frame if level == "developer" else developer_month_frame
    return build(db, project_ids, abandon_months)
