"""Context stages, run after the analysis stages on already-mined projects (adding them resets nothing):

- repo_info: repository metadata (archived, last push), so dead projects can be kept out of abandonment models;
- pr_outcomes: whether and by whom each pull request was merged (GraphQL, 100 PRs per request);
- post_exit: public GitHub activity elsewhere in the 12 months after a developer's last month here, which tells
  leaving the project apart from leaving open source (GraphQL contributionsCollection, one request per person).
"""

from datetime import datetime, timedelta

from sqlalchemy import delete, func, select

from ..github import GitHubError
from ..models import Alias, Developer, DeveloperMonth, Issue, ProjectMetric, utcnow
from ..pipeline import Stage, StageContext
from .github_items import ts

AFTER_EXIT_MONTHS = 12
# Shortest abandonment threshold for which post_exit has the data; candidates are checked once.
MIN_GAP_MONTHS = 6


class RepoInfoStage(Stage):
    name = "repo_info"
    label = "Repository info"
    description = "Archived flag, creation and last push dates (GitHub REST)."

    def run(self, ctx: StageContext):
        repo = ctx.github.get_optional(f"/repos/{ctx.project.full_name}")
        if repo is None:
            raise RuntimeError("repository not found on GitHub")
        info = {k: repo.get(k) for k in ("archived", "disabled", "fork", "created_at", "pushed_at", "stargazers_count")}
        ctx.db.execute(delete(ProjectMetric).where(ProjectMetric.project_id == ctx.project.id, ProjectMetric.key == "repo_info"))
        ctx.db.add(ProjectMetric(project_id=ctx.project.id, key="repo_info", value=info))
        ctx.progress(1, 1, "archived" if info["archived"] else f"last push {str(info['pushed_at'])[:10]}")


PR_OUTCOMES_QUERY = """
query($owner: String!, $name: String!, $cursor: String, $direction: OrderDirection = ASC) {
  repository(owner: $owner, name: $name) {
    pullRequests(first: 100, after: $cursor, orderBy: {field: CREATED_AT, direction: $direction}) {
      totalCount
      pageInfo { hasNextPage endCursor }
      nodes { number createdAt merged mergedAt mergedBy { login } }
    }
  }
}
"""


class PullOutcomesStage(Stage):
    name = "pr_outcomes"
    label = "Pull-request outcomes"
    description = "Merged or not, when and by whom, 100 PRs per GraphQL request."

    def run(self, ctx: StageContext):
        if not any(t.token for t in ctx.github.tokens):
            ctx.log("No GitHub token: pull-request outcomes need GraphQL, skipped", "warn")
            ctx.cursor["skipped"] = True
            return
        pid = ctx.project.id
        prs = {i.number: i for i in ctx.db.scalars(select(Issue).where(Issue.project_id == pid, Issue.is_pr))}
        aliases = dict(ctx.db.execute(select(Alias.key, Alias.id).where(Alias.project_id == pid, Alias.kind == "gh")).tuples().all())
        done = ctx.cursor.get("done", 0)
        since = ctx.project.since  # newest first, stop at the first PR older than the mined period
        while True:
            data = ctx.github.graphql(PR_OUTCOMES_QUERY, {"owner": ctx.project.owner, "name": ctx.project.name,
                                                          "cursor": ctx.cursor.get("gql_cursor"),
                                                          "direction": "DESC" if since else "ASC"})  # fmt: skip
            page = (data.get("repository") or {}).get("pullRequests")
            if page is None:
                raise RuntimeError("GraphQL: repository not found")
            reached_since = False
            for node in page["nodes"]:
                if since and ts(node["createdAt"]) < since:
                    reached_since = True
                    break
                pr = prs.get(node["number"])
                if pr is None:
                    continue
                pr.merged = bool(node["merged"])
                pr.merged_at = ts(node.get("mergedAt"))
                login = (node.get("mergedBy") or {}).get("login")
                # Only people already known to the project; an unknown merger stays None.
                pr.merged_by_alias_id = aliases.get(f"gh:{login.lower()}") if login else None
                done += 1
            ctx.cursor.update(gql_cursor=page["pageInfo"]["endCursor"] or ctx.cursor.get("gql_cursor"), done=done)
            ctx.progress(done, page["totalCount"], f"{done}/{page['totalCount']} pull requests")
            ctx.checkpoint()
            if reached_since or not page["pageInfo"]["hasNextPage"]:
                break
        merged = sum(1 for p in prs.values() if p.merged)
        ctx.log(f"Pull-request outcomes: {merged} of {len(prs)} merged")


BY_REPOSITORY = ("commitContributionsByRepository", "issueContributionsByRepository",
                 "pullRequestContributionsByRepository", "pullRequestReviewContributionsByRepository")  # fmt: skip


def contributions_query(fields=BY_REPOSITORY) -> str:
    parts = "\n".join(f"{f}(maxRepositories: 100) {{ repository {{ nameWithOwner }} contributions {{ totalCount }} }}" for f in fields)
    return f"""
query($login: String!, $from: DateTime!, $to: DateTime!) {{
  user(login: $login) {{ contributionsCollection(from: $from, to: $to) {{ restrictedContributionsCount {parts} }} }}
}}"""


CONTRIBUTIONS_QUERY = contributions_query()


def month_start(month: str) -> datetime:
    y, m = map(int, month.split("-"))
    return datetime(y, m, 1)


def add_months(d: datetime, n: int) -> datetime:
    y, m = divmod(d.month - 1 + n, 12)
    return d.replace(year=d.year + y, month=m + 1)


def elsewhere(collection: dict, repo: str) -> tuple[int, int]:
    """(contributions, distinct repositories) outside `repo` in a contributionsCollection."""
    total, repos = 0, set()
    for key in BY_REPOSITORY:
        for item in collection.get(key) or []:
            name = item["repository"]["nameWithOwner"]
            if name.lower() == repo.lower():
                continue
            total += item["contributions"]["totalCount"]
            repos.add(name.lower())
    return total, len(repos)


class PostExitStage(Stage):
    @staticmethod
    def fetch(ctx: StageContext, login: str, start: datetime, stop: datetime) -> dict | None:
        """`user` with a merged contributionsCollection, or None (no such account, or GitHub kept timing out).
        Very active accounts can time out the combined query on GitHub's side: then one field per query."""
        variables = {"login": login, "from": start.isoformat() + "Z", "to": stop.isoformat() + "Z"}
        try:
            return (ctx.github.graphql(CONTRIBUTIONS_QUERY, variables) or {}).get("user")
        except GitHubError:
            merged: dict = {}
            for field in BY_REPOSITORY:
                try:
                    user = (ctx.github.graphql(contributions_query((field,)), variables) or {}).get("user")
                except GitHubError:
                    ctx.log(f"Activity after leaving: GitHub timed out for one developer ({field})", "warn")
                    return None
                if user is None:
                    return None
                merged.update(user["contributionsCollection"])
            return {"contributionsCollection": merged}

    name = "post_exit"
    label = "Activity after leaving"
    description = "Public GitHub activity elsewhere in the 12 months after a developer's last month here (GraphQL)."

    def run(self, ctx: StageContext):
        if not any(t.token for t in ctx.github.tokens):
            ctx.log("No GitHub token: activity after leaving needs GraphQL, skipped", "warn")
            ctx.cursor["skipped"] = True
            return
        pid = ctx.project.id
        end = ctx.db.scalar(select(func.max(DeveloperMonth.month)).where(DeveloperMonth.project_id == pid))
        if end is None:
            return
        months = select(
            DeveloperMonth.developer_id, func.max(DeveloperMonth.month).label("last"), func.count().label("n")
        ).where(DeveloperMonth.project_id == pid).group_by(DeveloperMonth.developer_id).subquery()
        rows = ctx.db.execute(
            select(Developer, months.c.last)
            .join(months, months.c.developer_id == Developer.id)
            .where(~Developer.is_bot, Developer.login.is_not(None), months.c.n >= 2)
        ).all()
        # Candidates: could count as abandoners for any threshold >= MIN_GAP_MONTHS.
        todo = [(d, last) for d, last in rows if add_months(month_start(last), MIN_GAP_MONTHS) <= month_start(end)]
        now = utcnow()
        for n, (dev, last) in enumerate(todo, 1):
            start = add_months(month_start(last), 1)
            if dev.after_exit_from == start:
                continue  # already checked (resume)
            stop = min(add_months(start, AFTER_EXIT_MONTHS), now) - timedelta(seconds=1)
            user = self.fetch(ctx, dev.login, start, stop)
            dev.after_exit_from = start
            if user is None:
                dev.after_exit_contributions = dev.after_exit_repos = dev.after_exit_private = None
            else:
                c = user["contributionsCollection"]
                dev.after_exit_contributions, dev.after_exit_repos = elsewhere(c, ctx.project.full_name)
                dev.after_exit_private = c["restrictedContributionsCount"]
            if n % 10 == 0 or n == len(todo):
                ctx.progress(n, len(todo), f"{n}/{len(todo)} developers checked")
                ctx.checkpoint()
        unknown = sum(1 for d, _ in todo if d.after_exit_contributions is None)
        if unknown:
            ctx.log(f"Activity after leaving: {unknown} developers could not be checked (deleted account or GitHub timeout)", "warn")
        active = sum(1 for d, _ in todo if (d.after_exit_contributions or 0) + (d.after_exit_private or 0) > 0)
        ctx.log(f"Activity after leaving: {active} of {len(todo)} candidates active elsewhere within {AFTER_EXIT_MONTHS} months")
