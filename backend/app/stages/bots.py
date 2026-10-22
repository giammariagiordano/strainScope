import re

from sqlalchemy import select

from ..models import Alias
from ..pipeline import Stage, StageContext

KNOWN_BOTS = {
    "dependabot", "dependabot-preview", "renovate", "renovate-bot", "github-actions", "greenkeeper",
    "greenkeeperio-bot", "codecov", "codecov-io", "snyk-bot", "pre-commit-ci", "mergify", "allcontributors",
    "netlify", "vercel", "sonarcloud", "coveralls", "travis-ci", "k8s-ci-robot", "stale", "imgbot",
    "deepsource-autofix", "gitter-badger", "pyup-bot", "whitesource-bolt-for-github", "copilot",
    "web-flow", "github", "semantic-release-bot", "azure-pipelines", "codacy-badger", "lgtm-com",
}  # fmt: skip

BOT_PATTERN = re.compile(r"(\[bot\]|[-_.]bot$|^bot[-_.]|-robot$|^ci$)", re.IGNORECASE)


def looks_like_bot(alias: Alias) -> bool:
    if alias.gh_type == "Bot":
        return True
    candidates = [alias.login, alias.name, (alias.email or "").split("@")[0]]
    for value in filter(None, candidates):
        v = value.strip().lower()
        if v in KNOWN_BOTS or BOT_PATTERN.search(v):
            return True
    return False


class BotsStage(Stage):
    name = "bots"
    label = "Bot filter"
    description = "Flags bot accounts (GitHub Bot type, [bot] suffix, known bot names)."

    def run(self, ctx: StageContext):
        aliases = list(ctx.db.scalars(select(Alias).where(Alias.project_id == ctx.project.id)))
        bots = 0
        for a in aliases:
            a.is_bot = looks_like_bot(a)
            bots += a.is_bot
        ctx.progress(len(aliases), len(aliases), f"{bots} bot identities out of {len(aliases)}")
        ctx.log(f"Flagged {bots} bot identities")
