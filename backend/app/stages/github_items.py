"""Issues/PRs, comments and reviews from the GitHub REST API."""

from datetime import datetime

from sqlalchemy import func, select

from ..models import Comment, Issue
from ..pipeline import Stage, StageContext


def ts(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None) if value else None


def _issue_number(url: str | None) -> int | None:
    # e.g. https://api.github.com/repos/o/r/issues/12  or  .../pulls/12
    try:
        return int(url.rstrip("/").rsplit("/", 1)[1]) if url else None
    except ValueError:
        return None


def upsert_comment(ctx: StageContext, kind: str, gh_id: int, **fields):
    row = ctx.db.scalar(
        select(Comment).where(Comment.project_id == ctx.project.id, Comment.kind == kind, Comment.gh_id == gh_id)
    )
    if row is None:
        ctx.db.add(Comment(project_id=ctx.project.id, kind=kind, gh_id=gh_id, **fields))
    else:
        for k, v in fields.items():
            setattr(row, k, v)


class IssuesStage(Stage):
    name = "issues"
    label = "Issues & PRs"
    description = "All issues and pull requests, incrementally by last update."

    def run(self, ctx: StageContext):
        pid = ctx.project.id
        since = ctx.cursor.get("since")
        params = {"state": "all", "sort": "updated", "direction": "asc"}
        if since:
            params["since"] = since
        for _, items in ctx.github.paginate(f"/repos/{ctx.project.full_name}/issues", params):
            for it in items:
                fields = dict(
                    is_pr="pull_request" in it,
                    alias_id=ctx.gh_alias_id(it.get("user")),
                    title=it.get("title"),
                    body=it.get("body"),
                    state=it["state"],
                    created_at=ts(it["created_at"]),
                    updated_at=ts(it["updated_at"]),
                    closed_at=ts(it.get("closed_at")),
                )
                row = ctx.db.scalar(select(Issue).where(Issue.project_id == pid, Issue.number == it["number"]))
                if row is None:
                    ctx.db.add(Issue(project_id=pid, number=it["number"], **fields))
                else:
                    for k, v in fields.items():
                        setattr(row, k, v)
                ctx.cursor["since"] = it["updated_at"]
            count = ctx.db.scalar(select(func.count()).select_from(Issue).where(Issue.project_id == pid))
            ctx.progress(count, None, f"{count} issues/PRs, updated up to {ctx.cursor['since'][:10]}")
            ctx.checkpoint()
        ctx.log(f"Issues/PRs synced up to {ctx.cursor.get('since')}")


class CommentsStage(Stage):
    name = "comments"
    label = "Comments"
    description = "Issue/PR conversation comments and inline code-review comments."

    ENDPOINTS = {
        "issue_comment": ("issues/comments", "issue_url"),
        "review_comment": ("pulls/comments", "pull_request_url"),
    }

    def run(self, ctx: StageContext):
        pid = ctx.project.id
        for kind, (endpoint, url_field) in self.ENDPOINTS.items():
            since = ctx.cursor.get(kind)
            params = {"sort": "updated", "direction": "asc"}
            if since:
                params["since"] = since
            for _, items in ctx.github.paginate(f"/repos/{ctx.project.full_name}/{endpoint}", params):
                for it in items:
                    number = _issue_number(it.get(url_field))
                    if number is None:
                        continue
                    upsert_comment(
                        ctx, kind, it["id"],
                        issue_number=number,
                        alias_id=ctx.gh_alias_id(it.get("user")),
                        body=it.get("body"),
                        created_at=ts(it["created_at"]),
                        updated_at=ts(it.get("updated_at")),
                    )  # fmt: skip
                    ctx.cursor[kind] = it["updated_at"]
                count = ctx.db.scalar(select(func.count()).select_from(Comment).where(Comment.project_id == pid))
                ctx.progress(count, None, f"{count} comments ({kind.replace('_', ' ')}s up to {ctx.cursor[kind][:10]})")
                ctx.checkpoint()
        ctx.log("Comments synced")


class ReviewsStage(Stage):
    name = "reviews"
    label = "Reviews"
    description = "Pull-request reviews (approvals, change requests, review bodies), one PR at a time."

    def run(self, ctx: StageContext):
        pid = ctx.project.id
        prs = list(
            ctx.db.scalars(select(Issue.number).where(Issue.project_id == pid, Issue.is_pr).order_by(Issue.number))
        )
        last = ctx.cursor.get("last_pr", 0)
        todo = [n for n in prs if n > last]
        done = len(prs) - len(todo)
        for i, number in enumerate(todo, 1):
            # Fetch first, then write: keeps the write transaction short (SQLite has a single writer).
            reviews = [
                it
                for _, items in ctx.github.paginate(f"/repos/{ctx.project.full_name}/pulls/{number}/reviews")
                for it in items
                if it.get("submitted_at")  # skip pending reviews
            ]
            for it in reviews:
                upsert_comment(
                    ctx, "review", it["id"],
                    issue_number=number,
                    alias_id=ctx.gh_alias_id(it.get("user")),
                    body=it.get("body"),
                    review_state=it.get("state"),
                    created_at=ts(it["submitted_at"]),
                    updated_at=ts(it["submitted_at"]),
                )  # fmt: skip
            ctx.cursor["last_pr"] = number
            ctx.progress(done + i, len(prs), f"PR #{number}")
            ctx.checkpoint()
        ctx.progress(len(prs), len(prs), f"{len(prs)} pull requests")
