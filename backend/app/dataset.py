"""Analysis-ready tables built from the mined data (bots excluded, pseudonyms only)."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from .config import settings
from .models import Alias, Comment, Developer, DeveloperMonth, Issue, Project, ProjectMetric

LEVELS = ("project", "developer", "developer_month")


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
    Variable("we_share", "Belonging (we-talk)", "Share of first-person pronouns that are plural (we, our, let's) rather than singular (I, my) in the developer's messages: identification with the group (Pennebaker). Undefined without pronouns.", "share"),
    Variable("we_rate", "We-words per 100 words", "First-person plural pronouns per 100 words written.", "score"),
    Variable("i_rate", "I-words per 100 words", "First-person singular pronouns per 100 words written.", "score"),
    Variable("thanks_emitted_share", "Gratitude expressed", "Share of the developer's messages that thank or appreciate someone.", "share"),
    Variable("thanks_received", "Gratitude received", "Messages by other people (not bots) on the developer's issues/PRs that thank or appreciate."),
    Variable("thanks_received_share", "Gratitude received (share)", "Share of other people's messages on the developer's issues/PRs that thank or appreciate. Undefined without replies.", "share"),
    Variable("reactions_pos_per_item", "Positive reactions per item", "👍 ❤️ 🎉 🚀 reactions received per issue, PR or comment written by the developer.", "score"),
    Variable("reactions_neg_per_item", "Negative reactions per item", "👎 😕 reactions received per issue, PR or comment written by the developer.", "score"),
    Variable("anger_share", "Anger", "Share of the developer's messages expressing anger (EmoTxt).", "share"),
    Variable("joy_share", "Joy", "Share of the developer's messages expressing joy (EmoTxt).", "share"),
    Variable("love_share", "Love", "Share of the developer's messages expressing love (EmoTxt).", "share"),
    Variable("sadness_share", "Sadness", "Share of the developer's messages expressing sadness (EmoTxt).", "share"),
    Variable("fear_share", "Fear", "Share of the developer's messages expressing fear (EmoTxt).", "share"),
    Variable("surprise_share", "Surprise", "Share of the developer's messages expressing surprise (EmoTxt).", "share"),
    Variable("anger_received_share", "Anger received", "Share of other people's messages on the developer's issues/PRs expressing anger (EmoTxt).", "share"),
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
    Variable("we_share", "Belonging (we-talk)", "Share of first-person pronouns that are plural (we, our, let's) rather than singular (I, my) in all the developer's messages: identification with the group (Pennebaker). Undefined without pronouns.", "share"),
    Variable("we_rate", "We-words per 100 words", "First-person plural pronouns per 100 words written.", "score"),
    Variable("i_rate", "I-words per 100 words", "First-person singular pronouns per 100 words written.", "score"),
    Variable("thanks_emitted_share", "Gratitude expressed", "Share of the developer's messages that thank or appreciate someone.", "share"),
    Variable("thanks_received", "Gratitude received", "Messages by other people (not bots) on the developer's issues/PRs that thank or appreciate."),
    Variable("thanks_received_share", "Gratitude received (share)", "Share of other people's messages on the developer's issues/PRs that thank or appreciate. Undefined without replies.", "share"),
    Variable("reactions_pos_per_item", "Positive reactions per item", "👍 ❤️ 🎉 🚀 reactions received per issue, PR or comment written by the developer.", "score"),
    Variable("reactions_neg_per_item", "Negative reactions per item", "👎 😕 reactions received per issue, PR or comment written by the developer.", "score"),
    Variable("anger_share", "Anger", "Share of the developer's messages expressing anger (EmoTxt).", "share"),
    Variable("joy_share", "Joy", "Share of the developer's messages expressing joy (EmoTxt).", "share"),
    Variable("love_share", "Love", "Share of the developer's messages expressing love (EmoTxt).", "share"),
    Variable("sadness_share", "Sadness", "Share of the developer's messages expressing sadness (EmoTxt).", "share"),
    Variable("fear_share", "Fear", "Share of the developer's messages expressing fear (EmoTxt).", "share"),
    Variable("surprise_share", "Surprise", "Share of the developer's messages expressing surprise (EmoTxt).", "share"),
    Variable("anger_received_share", "Anger received", "Share of other people's messages on the developer's issues/PRs expressing anger (EmoTxt).", "share"),
    Variable("active_elsewhere_after", "Active elsewhere after leaving", "For developers who left (≥ 2 active months, last activity ≥ 6 months before the end): 1 if they made public GitHub contributions to other repositories (or private ones) in the 12 months after their last month here, 0 if not; empty if not checked.", "binary"),
    Variable("long_pauses", "Long pauses", "Pauses of at least the threshold after which the developer came back."),
    Variable("abandoned", "Abandoned", "1 if the developer has been inactive for at least the abandonment threshold at the end of observation. Right-censored: recent leavers count as 0.", "binary"),
]  # fmt: skip

PROJECT_VARS = [
    Variable("developers", "Developers", "Human developers with any activity."),
    Variable("sustained_developers", "Sustained developers", "Developers active in at least 6 months."),
    Variable("months_observed", "Months observed", "Months with any developer activity."),
    Variable("truck_factor", "Truck factor", "Truck factor at HEAD (Avelino et al.)."),
    Variable("core_size", "Core size", "Mean number of core developers per month."),
    Variable("abandonment_rate", "Abandonment rate", "Share of developers with ≥ 2 active months who abandoned (inactive ≥ threshold at the end).", "share"),
    Variable("moved_elsewhere_share", "Leavers active elsewhere", "Share of checked leavers with GitHub activity elsewhere in the 12 months after leaving.", "share"),
    Variable("off_hours_share", "Off-hours share", "Share of all commits on weekends or outside working hours.", "share"),
    Variable("negative_emitted_share", "Negativity", "Share of all messages with negative polarity (Senti4SD).", "share"),
    Variable("we_share", "Belonging (we-talk)", "Share of first-person pronouns that are plural, over all messages.", "share"),
    Variable("thanks_emitted_share", "Gratitude expressed", "Share of all messages that thank or appreciate someone.", "share"),
    Variable("thanks_received_share", "Gratitude in replies", "Share of replies on issues/PRs (by people other than the author) that thank or appreciate.", "share"),
    Variable("reactions_pos_per_item", "Positive reactions per item", "👍 ❤️ 🎉 🚀 per issue, PR or comment.", "score"),
    Variable("anger_share", "Anger", "Share of all messages expressing anger (EmoTxt).", "share"),
    Variable("joy_share", "Joy", "Share of all messages expressing joy (EmoTxt).", "share"),
    Variable("love_share", "Love", "Share of all messages expressing love (EmoTxt).", "share"),
    Variable("sadness_share", "Sadness", "Share of all messages expressing sadness (EmoTxt).", "share"),
    Variable("fear_share", "Fear", "Share of all messages expressing fear (EmoTxt).", "share"),
    Variable("surprise_share", "Surprise", "Share of all messages expressing surprise (EmoTxt).", "share"),
    Variable("pr_merge_rate", "PR merge rate", "Share of pull requests merged on GitHub (low values: mirror projects developed elsewhere).", "share"),
    Variable("response_days", "Days to first response", "Median days to the first response by another person on issues and PRs (unanswered = 30).", "score"),
    Variable("ai_pr_share", "AI PRs", "Share of pull requests with declared AI involvement (agent or disclosed).", "share"),
]  # fmt: skip

VARIABLES = {"project": PROJECT_VARS, "developer": DEVELOPER_VARS, "developer_month": DEVELOPER_MONTH_VARS}


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


RESPONSE_CAP_DAYS = 30


def response_times(db: Session, project_ids: list[int]) -> pd.DataFrame:
    """One row per issue/PR opened by a human: days until the first response by another human (a comment or
    review, or for a PR its merge by someone else; bots and the opener excluded). `unanswered` = no such response
    within RESPONSE_CAP_DAYS; `days` is then capped at RESPONSE_CAP_DAYS so medians stay defined.
    For PRs, `outcome_known` says whether the pr_outcomes stage fetched merges: without it a silently merged PR
    would look unanswered. `resolved_month` is the month of the merge or close (None while open)."""
    cols = ["project_id", "developer_id", "month", "is_pr", "outcome_known", "merged", "resolved_month", "days", "unanswered"]
    dev_of = dict(
        db.execute(
            select(Alias.id, Alias.developer_id)
            .join(Developer, Developer.id == Alias.developer_id)
            .where(Alias.project_id.in_(project_ids), ~Developer.is_bot)
        ).tuples().all()
    )  # fmt: skip
    issues = pd.DataFrame(
        db.execute(
            select(Issue.project_id, Issue.number, Issue.alias_id, Issue.is_pr, Issue.created_at, Issue.closed_at,
                   Issue.merged, Issue.merged_at, Issue.merged_by_alias_id)
            .where(Issue.project_id.in_(project_ids))
        ).all(),
        columns=["project_id", "number", "alias_id", "is_pr", "created_at", "closed_at", "merged", "merged_at", "merger_alias"],
    )  # fmt: skip
    comments = pd.DataFrame(
        db.execute(select(Comment.project_id, Comment.issue_number, Comment.alias_id, Comment.created_at).where(Comment.project_id.in_(project_ids))).all(),
        columns=["project_id", "number", "alias_id", "responded_at"],
    )  # fmt: skip
    if issues.empty:
        return pd.DataFrame(columns=cols)
    issues["developer_id"] = issues["alias_id"].map(dev_of)
    issues = issues.dropna(subset=["developer_id"])
    comments["responder"] = comments["alias_id"].map(dev_of)
    comments = comments.dropna(subset=["responder"])
    merges = issues.loc[issues["merged"].eq(True), ["project_id", "number", "merger_alias", "merged_at"]]
    merges = pd.DataFrame({"project_id": merges["project_id"], "number": merges["number"],
                           "responder": merges["merger_alias"].map(dev_of), "responded_at": merges["merged_at"]}).dropna()  # fmt: skip
    responses = pd.concat([comments[["project_id", "number", "responder", "responded_at"]], merges])
    m = issues.merge(responses, on=["project_id", "number"], how="left")
    m = m[m["responder"].isna() | (m["responder"] != m["developer_id"])]
    first = m.groupby(["project_id", "number"]).agg(
        developer_id=("developer_id", "first"), is_pr=("is_pr", "first"), created_at=("created_at", "first"),
        closed_at=("closed_at", "first"), merged=("merged", "first"), merged_at=("merged_at", "first"),
        responded_at=("responded_at", "min"),
    )
    days = (pd.to_datetime(first["responded_at"]) - first["created_at"]).dt.total_seconds() / 86400
    ym = lambda t: None if pd.isna(t) else f"{t.year:04d}-{t.month:02d}"  # noqa: E731
    resolved = [ym(ma) if mg is True else ym(c) for mg, ma, c in zip(first["merged"], first["merged_at"], first["closed_at"])]
    out = pd.DataFrame({
        "project_id": first.index.get_level_values(0),
        "developer_id": first["developer_id"].astype(int).to_numpy(),
        "month": first["created_at"].map(ym).to_numpy(),
        "is_pr": first["is_pr"].astype(bool).to_numpy(),
        "outcome_known": first["merged"].notna().to_numpy(),
        "merged": first["merged"].to_numpy(),
        "resolved_month": resolved,
        "unanswered": (days.isna() | (days > RESPONSE_CAP_DAYS)).to_numpy(),
        "days": days.clip(lower=0, upper=RESPONSE_CAP_DAYS).fillna(RESPONSE_CAP_DAYS).to_numpy(),
    })  # fmt: skip
    return out[cols]


PSYCH_COLUMNS = (
    DeveloperMonth.psych_messages, DeveloperMonth.words, DeveloperMonth.we_words, DeveloperMonth.i_words,
    DeveloperMonth.thanks_emitted, DeveloperMonth.replies_received, DeveloperMonth.thanks_received,
    DeveloperMonth.reacted_items, DeveloperMonth.reactions_pos_received, DeveloperMonth.reactions_neg_received,
    DeveloperMonth.emotion_messages, DeveloperMonth.anger_emitted, DeveloperMonth.joy_emitted, DeveloperMonth.love_emitted,
    DeveloperMonth.sadness_emitted, DeveloperMonth.fear_emitted, DeveloperMonth.surprise_emitted,
    DeveloperMonth.emotion_received, DeveloperMonth.anger_received,
)  # fmt: skip
PSYCH_KEYS = [c.key for c in PSYCH_COLUMNS]


def psych_ratios(df: pd.DataFrame) -> None:
    """Shares and rates of the psycholinguistic counts (app.psych), in place; undefined denominators give NaN."""
    nz = lambda c: df[c].replace(0, np.nan)  # noqa: E731
    df["we_share"] = df["we_words"] / (df["we_words"] + df["i_words"]).replace(0, np.nan)
    df["we_rate"] = 100 * df["we_words"] / nz("words")
    df["i_rate"] = 100 * df["i_words"] / nz("words")
    df["thanks_emitted_share"] = df["thanks_emitted"] / nz("psych_messages")
    df["thanks_received_share"] = df["thanks_received"] / nz("replies_received")
    df["reactions_pos_per_item"] = df["reactions_pos_received"] / nz("reacted_items")
    df["reactions_neg_per_item"] = df["reactions_neg_received"] / nz("reacted_items")
    for e in ("anger", "joy", "love", "sadness", "fear", "surprise"):
        df[f"{e}_share"] = df[f"{e}_emitted"] / nz("emotion_messages")
    df["anger_received_share"] = df["anger_received"] / nz("emotion_received")


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
            DeveloperMonth.threads, DeveloperMonth.toxic_threads, *PSYCH_COLUMNS,
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
    df[PSYCH_KEYS] = df[PSYCH_KEYS].astype(float)
    psych_ratios(df)
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
         "toxic_received", "negative_emitted", "negative_received", "threads", "toxic_threads", *PSYCH_KEYS,
         *(v.key for v in DEVELOPER_MONTH_VARS if v.key not in PSYCH_KEYS)]
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
        **{c: (c, lambda x: x.sum(min_count=1)) for c in PSYCH_KEYS},
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
    psych_ratios(out)
    overall = community_toxicity(db, out["project_id"].unique().tolist(), by_month=False)
    out["community_toxicity"] = out["project_id"].map(overall)
    checked = db.execute(
        select(Developer.id, Developer.after_exit_contributions, Developer.after_exit_private)
        .where(Developer.id.in_(out["developer_id"].unique().tolist()), Developer.after_exit_from.is_not(None))
    ).tuples().all()  # fmt: skip
    elsewhere = {d: float(pub + (priv or 0) > 0) for d, pub, priv in checked if pub is not None}
    out["active_elsewhere_after"] = out["developer_id"].map(elsewhere)
    return out[
        ["project_id", "developer_id", "project", "developer", "first_month", "last_month",
         *(v.key for v in DEVELOPER_VARS)]
    ]  # fmt: skip


def project_frame(
    db: Session, project_ids: list[int] | None = None, abandon_months: int | None = None
) -> pd.DataFrame:
    """One row per project: size, abandonment and the overall psychological climate (pooled over all messages)."""
    abandon_months = abandon_months or settings.abandon_months
    dm = developer_month_frame(db, project_ids, abandon_months)
    if dm.empty:
        return pd.DataFrame(columns=["project", *(v.key for v in PROJECT_VARS)])
    dev = developer_frame(db, project_ids, abandon_months)
    pids = [int(p) for p in dm["project_id"].unique()]
    sums = dm.groupby(["project_id", "project"])[
        ["commits", "off_hours_commits", "messages", "negative_emitted", *PSYCH_KEYS]
    ].sum(min_count=1).reset_index()  # fmt: skip
    psych_ratios(sums)
    sums["off_hours_share"] = sums["off_hours_commits"] / sums["commits"].replace(0, np.nan)
    sums["negative_emitted_share"] = sums["negative_emitted"] / sums["messages"].replace(0, np.nan)
    out = sums.set_index("project_id")
    out["developers"] = dev.groupby("project_id").size()
    out["sustained_developers"] = dev[dev["active_months"] >= 6].groupby("project_id").size()
    out["sustained_developers"] = out["sustained_developers"].fillna(0)
    out["months_observed"] = dm.groupby("project_id")["month"].nunique()
    out["core_size"] = dm.groupby(["project_id", "month"])["is_core"].sum().groupby("project_id").mean()
    stayers = dev[dev["active_months"] >= 2]
    out["abandonment_rate"] = stayers.groupby("project_id")["abandoned"].mean()
    out["moved_elsewhere_share"] = dev[dev["abandoned"] == 1].groupby("project_id")["active_elsewhere_after"].mean()
    tf = dict(db.execute(select(ProjectMetric.project_id, ProjectMetric.value)
                         .where(ProjectMetric.project_id.in_(pids), ProjectMetric.key == "truck_factor")).tuples().all())  # fmt: skip
    out["truck_factor"] = pd.Series({p: v.get("value") for p, v in tf.items()}, dtype=float)
    prs = pd.DataFrame(db.execute(select(Issue.project_id, Issue.merged, Issue.ai_source).where(Issue.project_id.in_(pids), Issue.is_pr)).all(),
                       columns=["project_id", "merged", "ai_source"])  # fmt: skip
    known = prs.dropna(subset=["merged"])
    out["pr_merge_rate"] = known.groupby("project_id")["merged"].mean().astype(float)
    out["ai_pr_share"] = prs.assign(ai=prs["ai_source"].notna()).groupby("project_id")["ai"].mean().astype(float)
    rt = response_times(db, pids)
    rt = rt[~rt["is_pr"] | rt["outcome_known"]]
    out["response_days"] = rt.groupby("project_id")["days"].median()
    return out.reset_index()[["project_id", "project", *(v.key for v in PROJECT_VARS)]]


def frame(
    db: Session, level: str, project_ids: list[int] | None = None, abandon_months: int | None = None
) -> pd.DataFrame:
    if level not in LEVELS:
        raise ValueError(f"unknown level {level}")
    build = {"project": project_frame, "developer": developer_frame, "developer_month": developer_month_frame}[level]
    return build(db, project_ids, abandon_months)
