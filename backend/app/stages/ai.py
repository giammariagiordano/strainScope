"""Declared AI involvement: agent accounts, AI co-author trailers in commits, and AI credits in PR descriptions.

Only *declared* use is visible, so every count is a lower bound. Patterns are deliberately narrow (tool-specific
accounts, e-mail domains and fixed phrases) to keep precision high: a person called Jules or a commit mentioning
"copilot" must not count. Validate precision on a manual sample before reporting.
"""

import re

from sqlalchemy import select

from ..models import Alias, Commit, Issue
from ..pipeline import Stage, StageContext
from .clone import git

# GitHub accounts of AI agents (login without "[bot]", lower case) -> (tool, role).
AGENT_ACCOUNTS = {
    "copilot-swe-agent": ("copilot", "author"),
    "copilot": ("copilot", "author"),
    "devin-ai-integration": ("devin", "author"),
    "chatgpt-codex-connector": ("codex", "author"),
    "openai-codex": ("codex", "author"),
    "claude": ("claude", "author"),
    "google-labs-jules": ("jules", "author"),
    "cursor": ("cursor", "author"),
    "openhands-agent": ("openhands", "author"),
    "openhands-ai": ("openhands", "author"),
    "sweep-ai": ("sweep", "author"),
    "amazon-q-developer": ("amazon-q", "author"),
    "codegen-sh": ("codegen", "author"),
    "factory-droid": ("factory", "author"),
    "copilot-pull-request-reviewer": ("copilot", "reviewer"),
    "coderabbitai": ("coderabbit", "reviewer"),
    "gemini-code-assist": ("gemini", "reviewer"),
    "ellipsis-dev": ("ellipsis", "reviewer"),
    "qodo-merge-pro": ("qodo", "reviewer"),
    "codiumai-pr-agent-pro": ("qodo", "reviewer"),
    "greptile-apps": ("greptile", "reviewer"),
    "sourcery-ai": ("sourcery", "reviewer"),
}

# Commit messages: co-author trailers identified by the tool's own e-mail address or account, and fixed
# "Generated with" lines. Checked in order; the first match names the tool.
COMMIT_PATTERNS = [
    ("claude", re.compile(r"^co-authored-by:.*<noreply@anthropic\.com>|generated with \[?claude code", re.I | re.M)),
    ("copilot", re.compile(r"^co-authored-by:.*<[^>]*\+copilot@users\.noreply\.github\.com>|^co-authored-by:.*copilot-swe-agent", re.I | re.M)),
    ("cursor", re.compile(r"^co-authored-by:.*<[^>]*@cursor\.(com|sh)>", re.I | re.M)),
    ("codex", re.compile(r"^co-authored-by:.*(<[^>]*@openai\.com>|chatgpt-codex-connector)", re.I | re.M)),
    ("devin", re.compile(r"^co-authored-by:.*devin-ai-integration", re.I | re.M)),
    ("jules", re.compile(r"^co-authored-by:.*google-labs-jules", re.I | re.M)),
    ("aider", re.compile(r"^co-authored-by:.*<[^>]*@aider\.chat>", re.I | re.M)),
    ("openhands", re.compile(r"^co-authored-by:.*<[^>]*@all-hands\.dev>", re.I | re.M)),
    ("amp", re.compile(r"^co-authored-by:.*<[^>]*@ampcode\.com>", re.I | re.M)),
    ("gemini", re.compile(r"^co-authored-by:.*gemini-code-assist", re.I | re.M)),
]

# PR descriptions written by people that credit an agent for the code (not tools that only write the
# description, such as Copilot PR summaries).
PR_PATTERNS = [
    ("claude", re.compile(r"generated with \[?claude code", re.I)),
    ("devin", re.compile(r"app\.devin\.ai/sessions|devin run:", re.I)),
    ("codex", re.compile(r"chatgpt\.com/codex", re.I)),
    ("cursor", re.compile(r"cursor\.com/agents|cursor\.com/background-agent", re.I)),
    ("jules", re.compile(r"jules\.google\.com/task", re.I)),
    ("openhands", re.compile(r"app\.all-hands\.dev", re.I)),
]


def agent_account(login: str | None) -> tuple[str, str] | None:
    if not login:
        return None
    return AGENT_ACCOUNTS.get(re.sub(r"\[bot\]$", "", login.strip(), flags=re.I).lower())


def commit_tool(message: str) -> str | None:
    return next((tool for tool, pattern in COMMIT_PATTERNS if pattern.search(message)), None)


def pr_tool(body: str | None) -> str | None:
    if not body:
        return None
    return next((tool for tool, pattern in PR_PATTERNS if pattern.search(body)), None)


class AISignalsStage(Stage):
    name = "ai_signals"
    label = "AI signals"
    description = "Declared AI involvement: agent accounts, AI co-author trailers in commits, AI credits in PRs."

    def run(self, ctx: StageContext):
        pid = ctx.project.id
        agents = {}
        for alias in ctx.db.scalars(select(Alias).where(Alias.project_id == pid)):
            hit = agent_account(alias.login)
            alias.ai_tool, alias.ai_role = hit if hit else (None, None)
            if hit:
                agents[alias.id] = hit
        ctx.progress(1, 3, f"{len(agents)} agent accounts")
        ctx.checkpoint()

        # Commit messages are not stored: read them from the local clone.
        log = git("log", "--all", "--format=%H%x1f%B%x1e", cwd=ctx.project.repo_path)
        credited = {}
        for record in log.split("\x1e"):
            sha, _, message = record.strip().partition("\x1f")
            if sha and (tool := commit_tool(message)):
                credited[sha] = tool
        commits = 0
        for c in ctx.db.scalars(select(Commit).where(Commit.project_id == pid)):
            author = agents.get(c.alias_id)
            c.ai_tool = credited.get(c.sha) or (author[0] if author and author[1] == "author" else None)
            commits += c.ai_tool is not None
        ctx.progress(2, 3, f"{commits} commits credit an AI tool")
        ctx.checkpoint()

        prs = {"agent": 0, "disclosed": 0}
        for pr in ctx.db.scalars(select(Issue).where(Issue.project_id == pid, Issue.is_pr)):
            author = agents.get(pr.alias_id)
            if author and author[1] == "author":
                pr.ai_tool, pr.ai_source = author[0], "agent"
            elif tool := pr_tool(pr.body):
                pr.ai_tool, pr.ai_source = tool, "disclosed"
            else:
                pr.ai_tool = pr.ai_source = None
            if pr.ai_source:
                prs[pr.ai_source] += 1
        ctx.progress(3, 3, f"{commits} AI commits; PRs: {prs['agent']} by agents, {prs['disclosed']} disclosed")
        ctx.log(f"AI signals: {len(agents)} agent accounts, {commits} commits, {prs['agent']} agent PRs, {prs['disclosed']} disclosed PRs")
