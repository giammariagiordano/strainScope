from sqlalchemy import func, select

from ..models import Alias, Commit
from ..pipeline import Stage, StageContext


class CommitAuthorsStage(Stage):
    name = "commit_authors"
    label = "Commit authors"
    description = "Links commit emails to GitHub logins via the commits API."

    def run(self, ctx: StageContext):
        pid = ctx.project.id
        total = ctx.db.scalar(select(func.count()).select_from(Commit).where(Commit.project_id == pid)) or 0
        start = ctx.cursor.get("page", 0) + 1
        linked = ctx.cursor.get("linked", 0)
        for page, items in ctx.github.paginate(f"/repos/{ctx.project.full_name}/commits", start_page=start):
            by_sha = {it["sha"]: it.get("author") for it in items if it.get("author")}
            if by_sha:
                rows = ctx.db.execute(
                    select(Commit.sha, Commit.alias_id).where(Commit.project_id == pid, Commit.sha.in_(by_sha))
                ).all()
                for sha, alias_id in rows:
                    user = by_sha[sha]
                    alias = ctx.db.get(Alias, alias_id)
                    if alias.login is None:
                        alias.login = user["login"]
                        alias.gh_type = user.get("type")
                        linked += 1
                    ctx.gh_alias_id(user)
            ctx.cursor.update(page=page, linked=linked)
            ctx.progress(min(page * 100, total), total, f"Page {page}, {linked} git identities linked")
            ctx.checkpoint()
        ctx.log(f"Linked {linked} git identities to GitHub logins")
