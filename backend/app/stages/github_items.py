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


def upsert_issue(ctx: StageContext, it: dict):
    fields = dict(
        is_pr="pull_request" in it,
        alias_id=ctx.gh_alias_id(it.get("user")),
        title=it.get("title"),
        body=it.get("body"),
        state=it["state"],
        created_at=ts(it["created_at"]),
        updated_at=ts(it["updated_at"]),
        closed_at=ts(it.get("closed_at")),
        comments_count=it.get("comments"),
    )
    row = ctx.db.scalar(select(Issue).where(Issue.project_id == ctx.project.id, Issue.number == it["number"]))
    if row is None:
        ctx.db.add(Issue(project_id=ctx.project.id, number=it["number"], **fields))
    else:
        for k, v in fields.items():
            setattr(row, k, v)


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
                upsert_issue(ctx, it)
                ctx.cursor["since"] = it["updated_at"]
            count = ctx.db.scalar(select(func.count()).select_from(Issue).where(Issue.project_id == pid))
            ctx.progress(count, None, f"{count} issues/PRs, updated up to {ctx.cursor['since'][:10]}")
            ctx.checkpoint()
        ctx.log(f"Issues/PRs synced up to {ctx.cursor.get('since')}")
        self.fill_missing(ctx)

    def fill_missing(self, ctx: StageContext):
        """Issues and PRs share one sequence 1..N: fetch individually every number the listing did not return.

        GitHub's listing endpoints occasionally return empty or truncated pages; this makes the result
        independent of them. Numbers that are deleted, transferred or used by discussions answer 404/410
        (or belong to another repository) and are skipped.
        """
        pid, repo = ctx.project.id, ctx.project.full_name
        newest = next(ctx.github.paginate(f"/repos/{repo}/issues", {"state": "all", "sort": "created", "direction": "desc"}), None)
        if not newest:
            return
        highest = max(it["number"] for it in newest[1])
        repo_url = ctx.github.get(f"/repos/{repo}").json()["url"]
        have = set(ctx.db.scalars(select(Issue.number).where(Issue.project_id == pid)))
        start = ctx.cursor.get("filled_upto", 0)
        missing = [n for n in range(start + 1, highest + 1) if n not in have]
        recovered = ctx.cursor.get("recovered", 0)
        for i, n in enumerate(missing, 1):
            it = ctx.github.get_optional(f"/repos/{repo}/issues/{n}")
            if it and it.get("repository_url") == repo_url:
                upsert_issue(ctx, it)
                recovered += 1
            ctx.cursor.update(filled_upto=n, recovered=recovered)
            if i % 25 == 0 or i == len(missing):
                ctx.progress(len(have) + recovered, highest, f"Checking {i}/{len(missing)} numbers missing from the listing")
                ctx.checkpoint()
        total = ctx.db.scalar(select(func.count()).select_from(Issue).where(Issue.project_id == pid))
        ctx.progress(total, total, f"{total} issues/PRs ({recovered} recovered individually)")
        if recovered:
            ctx.log(f"Recovered {recovered} issues/PRs that the listing did not return", "warn")


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
        self.fill_missing(ctx)

    def fill_missing(self, ctx: StageContext):
        """Compare each issue's comment count reported by GitHub with what was saved; refetch the gaps."""
        pid, repo = ctx.project.id, ctx.project.full_name
        saved = dict(
            ctx.db.execute(
                select(Comment.issue_number, func.count())
                .where(Comment.project_id == pid, Comment.kind == "issue_comment")
                .group_by(Comment.issue_number)
            ).all()
        )
        start = ctx.cursor.get("verified_upto", 0)
        expected = ctx.db.execute(
            select(Issue.number, Issue.comments_count)
            .where(Issue.project_id == pid, Issue.number > start, Issue.comments_count > 0)
            .order_by(Issue.number)
        ).all()
        gaps = [(n, c) for n, c in expected if saved.get(n, 0) < c]
        recovered = ctx.cursor.get("recovered", 0)
        for i, (n, _) in enumerate(gaps, 1):
            for _, items in ctx.github.paginate(f"/repos/{repo}/issues/{n}/comments"):
                for it in items:
                    upsert_comment(
                        ctx, "issue_comment", it["id"],
                        issue_number=n,
                        alias_id=ctx.gh_alias_id(it.get("user")),
                        body=it.get("body"),
                        created_at=ts(it["created_at"]),
                        updated_at=ts(it.get("updated_at")),
                    )  # fmt: skip
            after = ctx.db.scalar(
                select(func.count()).select_from(Comment).where(
                    Comment.project_id == pid, Comment.kind == "issue_comment", Comment.issue_number == n
                )
            )
            recovered += after - saved.get(n, 0)
            ctx.cursor.update(verified_upto=n, recovered=recovered)
            if i % 25 == 0 or i == len(gaps):
                ctx.progress(i, len(gaps), f"Refetching comments of {i}/{len(gaps)} incomplete threads")
                ctx.checkpoint()
        total = ctx.db.scalar(select(func.count()).select_from(Comment).where(Comment.project_id == pid))
        ctx.progress(total, total, f"{total} comments ({recovered} recovered from {len(gaps)} incomplete threads)")
        if recovered:
            ctx.log(f"Recovered {recovered} comments that the listing did not return", "warn")


REVIEWS_QUERY = """
query($owner: String!, $name: String!, $cursor: String) {
  repository(owner: $owner, name: $name) {
    pullRequests(first: 100, after: $cursor, orderBy: {field: CREATED_AT, direction: ASC}) {
      totalCount
      pageInfo { hasNextPage endCursor }
      nodes {
        number
        reviews(first: 100) {
          totalCount
          pageInfo { hasNextPage }
          nodes { databaseId state submittedAt body author { login __typename } }
        }
      }
    }
  }
}
"""


def graphql_user(author: dict | None) -> dict | None:
    """Map a GraphQL actor to the REST user shape: REST bot logins carry a `[bot]` suffix, GraphQL ones do not."""
    if not author or not author.get("login"):
        return None
    if author.get("__typename") == "Bot":
        return {"login": f"{author['login']}[bot]", "type": "Bot"}
    return {"login": author["login"], "type": author.get("__typename", "User")}


class ReviewsStage(Stage):
    name = "reviews"
    label = "Reviews"
    description = "Pull-request reviews (approvals, change requests, review bodies), 100 PRs per GraphQL request."

    def run(self, ctx: StageContext):
        if not any(t.token for t in ctx.github.tokens):
            ctx.log("No GitHub token: GraphQL is unavailable, fetching reviews one PR at a time", "warn")
            return self.run_rest(ctx)
        pid = ctx.project.id
        owner, name = ctx.project.owner, ctx.project.name
        seen: set[int] = set(ctx.cursor.get("seen", []))
        rest_fallback = ctx.cursor.get("rest_fallback", 0)
        while True:
            data = ctx.github.graphql(REVIEWS_QUERY, {"owner": owner, "name": name, "cursor": ctx.cursor.get("gql_cursor")})
            repo = data.get("repository")
            if repo is None:
                raise RuntimeError("GraphQL: repository not found")
            prs = repo["pullRequests"]
            for pr in prs["nodes"]:
                number = pr["number"]
                if pr["reviews"]["pageInfo"]["hasNextPage"]:
                    self.fetch_rest(ctx, number)  # > 100 reviews: rare, use REST for this PR
                    rest_fallback += 1
                else:
                    for rv in pr["reviews"]["nodes"]:
                        if not rv.get("submittedAt") or rv.get("databaseId") is None:
                            continue  # pending review
                        upsert_review(ctx, number, rv["databaseId"], graphql_user(rv.get("author")),
                                      rv.get("body"), rv.get("state"), rv["submittedAt"])  # fmt: skip
                seen.add(number)
            ctx.cursor.update(gql_cursor=prs["pageInfo"]["endCursor"] or ctx.cursor.get("gql_cursor"),
                              seen=sorted(seen), rest_fallback=rest_fallback)  # fmt: skip
            ctx.progress(len(seen), prs["totalCount"], f"{len(seen)}/{prs['totalCount']} pull requests (GraphQL)")
            ctx.checkpoint()
            if not prs["pageInfo"]["hasNextPage"]:
                break

        # PRs known from the issues listing but not returned by GraphQL: fetch them over REST.
        known = set(ctx.db.scalars(select(Issue.number).where(Issue.project_id == pid, Issue.is_pr)))
        missing = sorted(known - seen)
        for n in missing:
            self.fetch_rest(ctx, n)
            ctx.checkpoint()
        total = len(seen | known)
        ctx.progress(total, total, f"{total} pull requests (GraphQL; {rest_fallback + len(missing)} via REST)")
        if missing:
            ctx.log(f"{len(missing)} pull requests missing from GraphQL were fetched over REST", "warn")

    @staticmethod
    def fetch_rest(ctx: StageContext, number: int):
        path = f"/repos/{ctx.project.full_name}/pulls/{number}/reviews"
        for _, items in ctx.github.paginate(path, retry_empty=False):
            for it in items:
                if it.get("submitted_at"):  # skip pending reviews
                    upsert_review(ctx, number, it["id"], it.get("user"), it.get("body"), it.get("state"), it["submitted_at"])

    def run_rest(self, ctx: StageContext):
        """One request per PR; used only without a token."""
        pid = ctx.project.id
        prs = list(
            ctx.db.scalars(select(Issue.number).where(Issue.project_id == pid, Issue.is_pr).order_by(Issue.number))
        )
        last = ctx.cursor.get("last_pr", 0)
        todo = [n for n in prs if n > last]
        done = len(prs) - len(todo)
        for i, number in enumerate(todo, 1):
            self.fetch_rest(ctx, number)
            ctx.cursor["last_pr"] = number
            ctx.progress(done + i, len(prs), f"PR #{number}")
            ctx.checkpoint()
        ctx.progress(len(prs), len(prs), f"{len(prs)} pull requests")


def upsert_review(ctx: StageContext, number: int, gh_id: int, user: dict | None, body, state, submitted_at: str):
    upsert_comment(
        ctx, "review", gh_id,
        issue_number=number,
        alias_id=ctx.gh_alias_id(user),
        body=body,
        review_state=state,
        created_at=ts(submitted_at),
        updated_at=ts(submitted_at),
    )  # fmt: skip
