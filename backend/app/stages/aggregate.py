from collections import defaultdict
from datetime import timedelta

from sqlalchemy import delete, select

from ..config import settings
from ..models import Alias, Comment, Commit, Developer, DeveloperMonth, Issue
from ..pipeline import Stage, StageContext


def month_of(dt) -> str:
    return f"{dt.year:04d}-{dt.month:02d}"


def is_off_hours(local_dt) -> tuple[bool, bool]:
    """Return (off_hours, weekend) for a commit in the author's local time."""
    start, end = settings.work_hours
    weekend = local_dt.weekday() >= 5
    return weekend or not (start <= local_dt.hour < end), weekend


class AggregateStage(Stage):
    name = "aggregate"
    label = "Monthly aggregation"
    description = "Developer × month activity: commits, lines, off-hours share, issues, PRs, comments, reviews."

    def run(self, ctx: StageContext):
        pid = ctx.project.id
        dev_of = dict(ctx.db.execute(select(Alias.id, Alias.developer_id).where(Alias.project_id == pid)).all())
        rows: dict[tuple[int, str], dict] = defaultdict(lambda: defaultdict(int))
        first: dict[int, object] = {}
        last: dict[int, object] = {}

        def touch(dev_id, when):
            if dev_id is None or when is None:
                return None
            if dev_id not in first or when < first[dev_id]:
                first[dev_id] = when
            if dev_id not in last or when > last[dev_id]:
                last[dev_id] = when
            return rows[(dev_id, month_of(when))]

        commits = ctx.db.execute(
            select(Commit.alias_id, Commit.authored_at, Commit.tz_offset_min, Commit.additions, Commit.deletions)
            .where(Commit.project_id == pid)
        )  # fmt: skip
        for alias_id, authored, tz, add, dele in commits:
            r = touch(dev_of.get(alias_id), authored)
            if r is None:
                continue
            off, weekend = is_off_hours(authored + timedelta(minutes=tz))
            r["commits"] += 1
            r["additions"] += add
            r["deletions"] += dele
            r["off_hours_commits"] += off
            r["weekend_commits"] += weekend

        for alias_id, created, is_pr in ctx.db.execute(
            select(Issue.alias_id, Issue.created_at, Issue.is_pr).where(Issue.project_id == pid)
        ):
            if (r := touch(dev_of.get(alias_id), created)) is not None:
                r["prs_opened" if is_pr else "issues_opened"] += 1

        for alias_id, created, kind in ctx.db.execute(
            select(Comment.alias_id, Comment.created_at, Comment.kind).where(Comment.project_id == pid)
        ):
            if (r := touch(dev_of.get(alias_id), created)) is not None:
                r["reviews" if kind == "review" else "comments"] += 1

        ctx.db.execute(delete(DeveloperMonth).where(DeveloperMonth.project_id == pid))
        ctx.db.add_all(
            DeveloperMonth(project_id=pid, developer_id=dev_id, month=month, **values)
            for (dev_id, month), values in rows.items()
        )
        for dev in ctx.db.scalars(select(Developer).where(Developer.project_id == pid)):
            dev.first_activity = first.get(dev.id)
            dev.last_activity = last.get(dev.id)
        ctx.progress(len(rows), len(rows), f"{len(rows)} developer-months")
        ctx.log(f"Aggregated {len(rows)} developer-months")
