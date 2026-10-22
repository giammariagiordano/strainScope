"""Pseudonymized dataset export: one ZIP with CSV tables, a codebook and run metadata."""

import io
import json
import zipfile
from datetime import datetime, timezone

import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .config import settings
from .dataset import DEVELOPER_MONTH_VARS, DEVELOPER_VARS, developer_frame, developer_month_frame
from .survival import COVARIATES as SURVIVAL_COVARIATES
from .survival import build_intervals
from .models import Comment, Commit, Developer, Issue, Project, ProjectMetric

ID_COLUMNS = {
    "project": "Repository (owner/name).",
    "developer": "Stable pseudonym of the developer within the project (salted hash).",
    "month": "Calendar month, YYYY-MM.",
    "first_month": "First month with recorded activity.",
    "last_month": "Last month with recorded activity.",
}


def projects_frame(db: Session, project_ids: list[int] | None) -> pd.DataFrame:
    q = select(Project).order_by(Project.owner, Project.name)
    if project_ids:
        q = q.where(Project.id.in_(project_ids))
    rows = []
    for p in db.scalars(q):
        tf = db.scalar(select(ProjectMetric).where(ProjectMetric.project_id == p.id, ProjectMetric.key == "truck_factor"))

        def count(model, *where, _pid=p.id):
            return db.scalar(select(func.count()).select_from(model).where(model.project_id == _pid, *where))

        rows.append(
            {
                "project": p.full_name,
                "url": p.url,
                "pipeline_status": p.status,
                "mined_at": p.finished_at.isoformat() + "Z" if p.finished_at else None,
                "commits": count(Commit),
                "issues": count(Issue, ~Issue.is_pr),
                "pull_requests": count(Issue, Issue.is_pr),
                "comments": count(Comment, Comment.kind != "review"),
                "reviews": count(Comment, Comment.kind == "review"),
                "developers": count(Developer, ~Developer.is_bot),
                "bot_accounts": count(Developer, Developer.is_bot),
                "truck_factor": tf.value["value"] if tf else None,
                "truck_factor_files": tf.value["files"] if tf else None,
            }
        )
    return pd.DataFrame(rows)


def codebook() -> str:
    lines = [
        "# StrainScope dataset codebook",
        "",
        "All tables exclude bot accounts. Developers are identified only by pseudonyms;",
        "no names, logins, emails or message text are included.",
        "Pseudonymized data is still personal data under the GDPR: do not attempt re-identification.",
        "",
        "## projects.csv",
        "One row per repository: size, pipeline status and truck factor (Avelino et al., ICPC 2016).",
        "",
        "## developers.csv — one row per developer × project",
        "",
        "| Column | Description |",
        "|---|---|",
    ]
    lines += [f"| `{k}` | {d} |" for k, d in ID_COLUMNS.items() if k != "month"]
    lines += [f"| `{v.key}` | {v.description} |" for v in DEVELOPER_VARS]
    lines += [
        "",
        "## developer_months.csv — one row per active developer × project × month",
        "",
        "| Column | Description |",
        "|---|---|",
    ]
    lines += [f"| `{k}` | {d} |" for k, d in ID_COLUMNS.items() if k in ("project", "developer", "month")]
    lines += [
        "| `truck_factor_member` | 1 if the developer is in the project's truck-factor set (project-level, repeated). |",
        "| `off_hours_commits` | Commits on weekends or outside working hours. |",
        "| `weekend_commits` | Commits on Saturday or Sunday. |",
        "| `inactive_months_at_end` | Months between the developer's last activity and the project's last observed month. |",
    ]
    lines += [f"| `{v.key}` | {v.description} |" for v in DEVELOPER_MONTH_VARS]
    lines += [
        "",
        "## survival_intervals.csv — counting-process table for time-to-abandonment models",
        "",
        "One row per ACTIVE developer-month at risk, (start, stop] in calendar months since entry; months of a",
        "pause are not at risk. `event` = 1 in the developer's last active month if they never come back",
        "(abandonment). For a model of long pauses instead, use `event_type != ''` as the event (recurrent:",
        "`break_return` = pause of at least the threshold followed by a return). Entry = first active month (not at risk itself);",
        "covariates summarize the previous 3 calendar months (lagged). Months later than the project's last",
        "observed month minus the threshold are censored. One-off contributors have no time at risk.",
        "The subject is project + developer (pseudonyms are unique only within a project). Fit with e.g. lifelines",
        "`CoxTimeVaryingFitter` (id = project + developer, strata = project) or R",
        "`coxph(Surv(start, stop, event) ~ ... + strata(project) + cluster(paste(project, developer)))`.",
        "",
        "| Column | Description |",
        "|---|---|",
        "| `start`, `stop` | Interval bounds in months since entry. |",
        "| `episode` | Always 0 in this export (abandonment is a single event). |",
        "| `event` | 1 if the developer stops here and never comes back (inactive ≥ threshold until the end of data). |",
        "| `event_type` | `break_return`, `abandonment` (no return) or empty. |",
    ]
    lines += [f"| `{c.key}` | {c.description} |" for c in SURVIVAL_COVARIATES]
    lines += [
        "",
        "## Parameters",
        f"- Working hours: {settings.work_hours[0]}:00–{settings.work_hours[1]}:00, Monday–Friday, author's local time.",
        f"- Core threshold: {settings.core_threshold:.0%} of cumulative eigenvector centrality (Joblin et al., ICSE 2017).",
        f"- Network window: {settings.network_window_months} months, file co-edits + issue/PR thread co-participation.",
        f"- Abandonment threshold: {settings.abandon_months} months of inactivity before the project's last observed month.",
        "  Right-censored: developers who left less than the threshold before the end of observation count as not abandoned.",
        "",
        "See the project README for known limitations and deviations from the original papers.",
    ]
    return "\n".join(lines) + "\n"


def build_zip(db: Session, project_ids: list[int] | None) -> bytes:
    projects = projects_frame(db, project_ids)
    devs = developer_frame(db, project_ids).drop(columns=["project_id", "developer_id"], errors="ignore")
    months = developer_month_frame(db, project_ids).drop(columns=["project_id", "developer_id"], errors="ignore")
    intervals, _ = build_intervals(db, project_ids, settings.abandon_months, window=3, min_active_months=1, outcome="abandonment")
    # `person` holds the GitHub login used for clustering: never exported.
    intervals = intervals.drop(columns=["sid", "person"], errors="ignore")
    meta = {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "tool": "StrainScope",
        "projects": projects["project"].tolist() if not projects.empty else [],
        "rows": {
            "projects": len(projects),
            "developers": len(devs),
            "developer_months": len(months),
            "survival_intervals": len(intervals),
        },
        "parameters": {
            "work_hours": list(settings.work_hours),
            "core_threshold": settings.core_threshold,
            "network_window_months": settings.network_window_months,
            "abandon_months": settings.abandon_months,
        },
    }
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("projects.csv", projects.to_csv(index=False))
        z.writestr("developers.csv", devs.to_csv(index=False, float_format="%.6g"))
        z.writestr("developer_months.csv", months.to_csv(index=False, float_format="%.6g"))
        z.writestr("survival_intervals.csv", intervals.to_csv(index=False, float_format="%.6g"))
        z.writestr("codebook.md", codebook())
        z.writestr("metadata.json", json.dumps(meta, indent=2))
    return buf.getvalue()
