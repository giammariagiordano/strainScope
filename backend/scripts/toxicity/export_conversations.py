"""Export StrainScope's conversations for LLM labelling (docs/TOXICITY.md, ASE pipeline).

Run from backend/ with the backend interpreter:
    PYTHONPATH=. .venv/bin/python scripts/toxicity/export_conversations.py

Output: external/toxicity-gh/work/strainscope_conversations.parquet, one row per issue/PR with at least one human
message, built exactly as the conversation_toxicity stage builds them (app.stages.text.conversation_text).
"""

from pathlib import Path

import pandas as pd
from sqlalchemy import select

import app.pipeline  # noqa: F401  (registers the stages, avoiding circular imports)
from app.config import settings
from app.db import SessionLocal
from app.models import Alias, Comment, Issue, Project
from app.stages.text import conversation_text

OUT = Path(settings.external_dir) / "toxicity-gh" / "work" / "strainscope_conversations.parquet"

rows = []
with SessionLocal() as db:
    for project in db.scalars(select(Project).order_by(Project.id)):
        bots = set(db.scalars(select(Alias.id).where(Alias.project_id == project.id, Alias.is_bot)))
        comments: dict[int, list] = {}
        for c in db.scalars(
            select(Comment).where(Comment.project_id == project.id, Comment.kind == "issue_comment").order_by(Comment.created_at)
        ):
            comments.setdefault(c.issue_number, []).append(c)
        for issue in db.scalars(select(Issue).where(Issue.project_id == project.id).order_by(Issue.number)):
            text = conversation_text(issue, comments.get(issue.number, []), bots)
            if text:
                rows.append({"id": issue.id, "project_id": project.id, "repo": project.full_name, "number": issue.number,
                             "is_pr": issue.is_pr, "created_at": issue.created_at, "text": text})  # fmt: skip

df = pd.DataFrame(rows)
OUT.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(OUT)
chars = df.text.str.len()
print(f"{len(df)} conversations from {df.repo.nunique()} repositories -> {OUT}")
print(f"chars: median {chars.median():.0f}, p90 {chars.quantile(0.9):.0f}, p99 {chars.quantile(0.99):.0f}, max {chars.max()}")
