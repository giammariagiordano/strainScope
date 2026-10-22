import re
import shutil
from collections import defaultdict
from datetime import timedelta
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import case, delete, func, select, update

from .config import BACKEND_DIR, settings
from .db import init_db, session_scope
from .models import (
    AppState,
    Comment,
    Commit,
    Developer,
    DeveloperMonth,
    Issue,
    Log,
    Project,
    ProjectMetric,
    StageRun,
    utcnow,
)
from . import analysis, survival
from .dataset import LEVELS, VARIABLES, frame
from .export import build_zip
from .github import read_rate_state
from .pipeline import STAGE_NAMES, ProjectRef, ensure_stage_runs, registry
from .stages.aggregate import is_off_hours

app = FastAPI(title="StrainScope API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

STAGES = {s.name: s for s in registry()}
REPO_RE = re.compile(r"^(?:https?://)?(?:www\.)?(?:github\.com/)?([\w.-]+)/([\w.-]+?)(?:\.git)?/?$")


@app.on_event("startup")
def _startup():
    init_db()


def parse_repo(url: str) -> tuple[str, str]:
    m = REPO_RE.match(url.strip())
    if not m:
        raise HTTPException(422, f"Not a GitHub repository URL: {url}")
    return m.group(1), m.group(2)


def month_range(first: str, last: str) -> list[str]:
    y, m = map(int, first.split("-"))
    out = []
    while f"{y:04d}-{m:02d}" <= last:
        out.append(f"{y:04d}-{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def dev_name(d: Developer) -> str:
    return (d.display_name or d.login or d.pseudonym) if settings.show_real_identities else d.pseudonym


# -- projects -------------------------------------------------------------------------


class NewProjects(BaseModel):
    urls: list[str]


def project_summary(s, p: Project) -> dict:
    runs = list(s.scalars(select(StageRun).where(StageRun.project_id == p.id).order_by(StageRun.position)))
    done = sum(r.status == "done" for r in runs)
    current = next((r for r in runs if r.stage == p.current_stage), None)
    return {
        "id": p.id,
        "full_name": p.full_name,
        "url": p.url,
        "status": p.status,
        "pause_requested": p.pause_requested,
        "current_stage": p.current_stage,
        "current_stage_label": STAGES[p.current_stage].label if p.current_stage in STAGES else None,
        "current_message": current.message if current else None,
        "stages_done": done,
        "stages_total": len(STAGE_NAMES),
        "error": p.error,
        "created_at": p.created_at.isoformat() + "Z",
        "finished_at": p.finished_at.isoformat() + "Z" if p.finished_at else None,
        "counts": {
            "commits": s.scalar(select(func.count()).select_from(Commit).where(Commit.project_id == p.id)),
            "issues": s.scalar(select(func.count()).select_from(Issue).where(Issue.project_id == p.id)),
            "developers": s.scalar(
                select(func.count()).select_from(Developer).where(Developer.project_id == p.id, ~Developer.is_bot)
            ),
        },
    }


def get_project(s, pid: int) -> Project:
    p = s.get(Project, pid)
    if p is None:
        raise HTTPException(404, "Project not found")
    return p


@app.post("/api/projects")
def create_projects(body: NewProjects):
    created, skipped = [], []
    with session_scope() as s:
        for url in [u for u in body.urls if u.strip()]:
            owner, name = parse_repo(url)
            existing = s.scalar(select(Project).where(Project.owner == owner, Project.name == name))
            if existing:
                skipped.append(existing.full_name)
                continue
            p = Project(owner=owner, name=name, url=f"https://github.com/{owner}/{name}", status="queued")
            s.add(p)
            s.flush()
            ensure_stage_runs(s, p.id)
            s.add(Log(project_id=p.id, message="Project added to the queue"))
            created.append(p.full_name)
    return {"created": created, "skipped": skipped}


@app.get("/api/projects")
def list_projects():
    with session_scope() as s:
        return [project_summary(s, p) for p in s.scalars(select(Project).order_by(Project.created_at.desc()))]


@app.get("/api/projects/{pid}")
def project_detail(pid: int):
    with session_scope() as s:
        p = get_project(s, pid)
        runs = s.scalars(select(StageRun).where(StageRun.project_id == pid).order_by(StageRun.position))
        return {
            **project_summary(s, p),
            "stages": [
                {
                    "name": r.stage,
                    "label": STAGES[r.stage].label if r.stage in STAGES else r.stage,
                    "description": STAGES[r.stage].description if r.stage in STAGES else "",
                    "status": r.status,
                    "done": r.done,
                    "total": r.total,
                    "message": r.message,
                    "started_at": r.started_at.isoformat() + "Z" if r.started_at else None,
                    "finished_at": r.finished_at.isoformat() + "Z" if r.finished_at else None,
                }
                for r in runs
            ],
        }


@app.post("/api/projects/{pid}/pause")
def pause(pid: int):
    with session_scope() as s:
        p = get_project(s, pid)
        if p.status == "running":
            p.pause_requested = True
        elif p.status == "queued":
            p.status = "paused"
        s.add(Log(project_id=pid, message="Pause requested"))
    return {"ok": True}


@app.post("/api/projects/{pid}/resume")
def resume(pid: int):
    with session_scope() as s:
        p = get_project(s, pid)
        if p.status == "running":
            p.pause_requested = False
            return {"ok": True}
        s.execute(
            update(StageRun)
            .where(StageRun.project_id == pid, StageRun.status.in_(["failed", "paused", "running"]))
            .values(status="pending")
        )
        p.status, p.pause_requested, p.error = "queued", False, None
        s.add(Log(project_id=pid, message="Resumed"))
    return {"ok": True}


@app.post("/api/projects/{pid}/rerun/{stage}")
def rerun(pid: int, stage: str):
    if stage not in STAGE_NAMES:
        raise HTTPException(404, "Unknown stage")
    with session_scope() as s:
        p = get_project(s, pid)
        if p.status == "running":
            raise HTTPException(409, "Pause the project before re-running a stage")
        later = STAGE_NAMES[STAGE_NAMES.index(stage) :]
        s.execute(
            update(StageRun)
            .where(StageRun.project_id == pid, StageRun.stage.in_(later))
            .values(status="pending", cursor={}, done=0, total=None, message=None, started_at=None, finished_at=None)
        )
        p.status, p.pause_requested, p.error, p.finished_at = "queued", False, None, None
        s.add(Log(project_id=pid, message=f"Re-running from {STAGES[stage].label}"))
    return {"ok": True}


@app.delete("/api/projects/{pid}")
def delete_project(pid: int):
    with session_scope() as s:
        p = get_project(s, pid)
        if p.status == "running":
            raise HTTPException(409, "Pause the project before deleting it")
        ref = ProjectRef(p.id, p.owner, p.name)
        s.execute(delete(Project).where(Project.id == pid))
    shutil.rmtree(ref.repo_path, ignore_errors=True)
    return {"ok": True}


@app.get("/api/projects/{pid}/logs")
def logs(pid: int, limit: int = 200):
    with session_scope() as s:
        rows = s.scalars(select(Log).where(Log.project_id == pid).order_by(Log.id.desc()).limit(limit))
        return [
            {"id": l.id, "ts": l.ts.isoformat() + "Z", "level": l.level, "stage": l.stage, "message": l.message}
            for l in rows
        ]


# -- metrics --------------------------------------------------------------------------


@app.get("/api/projects/{pid}/overview")
def overview(pid: int):
    with session_scope() as s:
        p = get_project(s, pid)
        devs = {d.id: d for d in s.scalars(select(Developer).where(Developer.project_id == pid))}
        humans = {i for i, d in devs.items() if not d.is_bot}

        def count(model, *where):
            return s.scalar(select(func.count()).select_from(model).where(model.project_id == pid, *where))

        tf = s.scalar(select(ProjectMetric).where(ProjectMetric.project_id == pid, ProjectMetric.key == "truck_factor"))
        tf_value = None
        if tf:
            tf_value = {
                **tf.value,
                "members": [
                    {"id": m, "name": dev_name(devs[m]), "files": devs[m].files_authored}
                    for m in tf.value.get("members", [])
                    if m in devs
                ],
            }

        monthly: dict[str, dict] = defaultdict(lambda: defaultdict(float))
        for r in s.scalars(select(DeveloperMonth).where(DeveloperMonth.project_id == pid)):
            if r.developer_id not in humans:
                continue
            m = monthly[r.month]
            m["commits"] += r.commits
            m["off_hours_commits"] += r.off_hours_commits
            m["issues"] += r.issues_opened
            m["prs"] += r.prs_opened
            m["comments"] += r.comments
            m["reviews"] += r.reviews
            m["active"] += 1
            m["core"] += r.is_core
        keys = ("commits", "off_hours_commits", "issues", "prs", "comments", "reviews", "active", "core")
        months = month_range(min(monthly), max(monthly)) if monthly else []
        series = [
            {
                "month": month,
                **{k: int(monthly[month][k]) if month in monthly else 0 for k in keys},
                "off_hours_share": (
                    round(monthly[month]["off_hours_commits"] / monthly[month]["commits"], 3)
                    if month in monthly and monthly[month]["commits"]
                    else None
                ),
            }
            for month in months
        ]

        heat = [[0] * 24 for _ in range(7)]  # weekday x local hour
        off = 0
        for authored, tz in s.execute(
            select(Commit.authored_at, Commit.tz_offset_min).where(Commit.project_id == pid)
        ):
            local = authored + timedelta(minutes=tz)
            heat[local.weekday()][local.hour] += 1
            off += is_off_hours(local)[0]
        total_commits = sum(map(sum, heat))

        return {
            "project": project_summary(s, p),
            "totals": {
                "commits": total_commits,
                "issues": count(Issue, ~Issue.is_pr),
                "prs": count(Issue, Issue.is_pr),
                "comments": count(Comment, Comment.kind != "review"),
                "reviews": count(Comment, Comment.kind == "review"),
                "developers": len(humans),
                "bots": len(devs) - len(humans),
                "off_hours_share": round(off / total_commits, 3) if total_commits else None,
            },
            "truck_factor": tf_value,
            "monthly": series,
            "heatmap": heat,
            "work_hours": list(settings.work_hours),
            "real_identities": settings.show_real_identities,
        }


@app.get("/api/projects/{pid}/developers")
def developers(pid: int, include_bots: bool = False):
    with session_scope() as s:
        get_project(s, pid)
        agg = {
            row.developer_id: row
            for row in s.execute(
                select(
                    DeveloperMonth.developer_id,
                    func.count().label("months"),
                    func.sum(DeveloperMonth.commits).label("commits"),
                    func.sum(DeveloperMonth.additions + DeveloperMonth.deletions).label("churn"),
                    func.sum(DeveloperMonth.off_hours_commits).label("off_hours"),
                    func.sum(DeveloperMonth.issues_opened + DeveloperMonth.prs_opened).label("opened"),
                    func.sum(DeveloperMonth.comments).label("comments"),
                    func.sum(DeveloperMonth.reviews).label("reviews"),
                    func.sum(case((DeveloperMonth.is_core, 1), else_=0)).label("core_months"),
                    func.max(DeveloperMonth.centrality).label("max_centrality"),
                )
                .where(DeveloperMonth.project_id == pid)
                .group_by(DeveloperMonth.developer_id)
            )
        }
        last_month = s.scalar(select(func.max(DeveloperMonth.month)).where(DeveloperMonth.project_id == pid))
        out = []
        for d in s.scalars(select(Developer).where(Developer.project_id == pid)):
            if d.is_bot and not include_bots:
                continue
            a = agg.get(d.id)
            if a is None:
                continue
            out.append(
                {
                    "id": d.id,
                    "name": dev_name(d),
                    "pseudonym": d.pseudonym,
                    "is_bot": d.is_bot,
                    "truck_factor_member": d.truck_factor_member,
                    "files_authored": d.files_authored,
                    "first_activity": d.first_activity.isoformat() + "Z" if d.first_activity else None,
                    "last_activity": d.last_activity.isoformat() + "Z" if d.last_activity else None,
                    "active_months": a.months,
                    "core_months": int(a.core_months or 0),
                    "commits": int(a.commits or 0),
                    "churn": int(a.churn or 0),
                    "off_hours_share": round(a.off_hours / a.commits, 3) if a.commits else None,
                    "opened": int(a.opened or 0),
                    "comments": int(a.comments or 0),
                    "reviews": int(a.reviews or 0),
                    "max_centrality": round(a.max_centrality or 0, 4),
                }
            )
        out.sort(key=lambda r: (-r["core_months"], -r["commits"]))
        return {"last_month": last_month, "developers": out}


EMPTY_MONTH = dict.fromkeys(
    ("commits", "churn", "off_hours_commits", "weekend_commits", "issues_opened", "prs_opened", "comments", "reviews", "degree"),
    0,
) | {"centrality": None, "is_core": False}  # not measured when inactive


@app.get("/api/projects/{pid}/developers/{dev_id}")
def developer_timeline(pid: int, dev_id: int):
    with session_scope() as s:
        d = s.get(Developer, dev_id)
        if d is None or d.project_id != pid:
            raise HTTPException(404, "Developer not found")
        rows = {
            r.month: r
            for r in s.scalars(select(DeveloperMonth).where(DeveloperMonth.developer_id == dev_id))
        }
        months = month_range(min(rows), max(rows)) if rows else []
        return {
            "id": d.id,
            "name": dev_name(d),
            "truck_factor_member": d.truck_factor_member,
            "files_authored": d.files_authored,
            "months": [
                {
                    "month": r.month,
                    "active": True,
                    "commits": r.commits,
                    "churn": r.additions + r.deletions,
                    "off_hours_commits": r.off_hours_commits,
                    "weekend_commits": r.weekend_commits,
                    "issues_opened": r.issues_opened,
                    "prs_opened": r.prs_opened,
                    "comments": r.comments,
                    "reviews": r.reviews,
                    "centrality": round(r.centrality, 4),
                    "degree": r.degree,
                    "is_core": r.is_core,
                }
                if (r := rows.get(month))
                else {"month": month, "active": False, **EMPTY_MONTH}
                for month in months
            ],
        }


@app.get("/api/status")
def status():
    with session_scope() as s:
        worker = s.get(AppState, "worker")
        alive = False
        if worker:
            from datetime import datetime

            alive = utcnow() - datetime.fromisoformat(worker.value["alive_at"]) < timedelta(
                seconds=settings.heartbeat_timeout_s
            )
        running = s.scalar(select(func.count()).select_from(Project).where(Project.status == "running"))
        return {
            "worker_alive": bool(alive or running),
            "github": read_rate_state(),
            "tokens_configured": len(settings.github_tokens),
            "real_identities": settings.show_real_identities,
        }


# -- export -----------------------------------------------------------------------


def parse_ids(project_ids: str | None) -> list[int] | None:
    if not project_ids:
        return None
    try:
        return [int(x) for x in project_ids.split(",") if x.strip()]
    except ValueError:
        raise HTTPException(422, "project_ids must be a comma-separated list of integers")


@app.get("/api/export")
def export(project_ids: str | None = None):
    """ZIP with projects.csv, developers.csv, developer_months.csv, codebook.md, metadata.json."""
    ids = parse_ids(project_ids)
    with session_scope() as s:
        if ids and len(ids) == 1:
            p = get_project(s, ids[0])
            name = f"strainscope_{p.owner}_{p.name}.zip"
        else:
            name = "strainscope_dataset.zip"
        data = build_zip(s, ids)
    return Response(
        data, media_type="application/zip", headers={"Content-Disposition": f'attachment; filename="{name}"'}
    )


# -- analysis ---------------------------------------------------------------------


@app.get("/api/analysis/variables")
def variables():
    return {
        "defaults": {"abandon_months": settings.abandon_months},
        "levels": {
            level: [{"key": v.key, "label": v.label, "description": v.description, "kind": v.kind} for v in vs]
            for level, vs in VARIABLES.items()
        },
    }


class AnalysisQuery(BaseModel):
    project_ids: list[int] | None = None
    level: str = "developer"
    min_active_months: int = 1
    abandon_months: int = settings.abandon_months


class CorrelationQuery(AnalysisQuery):
    method: str = "spearman"
    variables: list[str]
    controls: list[str] = []


class ScatterQuery(AnalysisQuery):
    x: str
    y: str


def analysis_frame(s, q: AnalysisQuery, needed: list[str]):
    if q.level not in LEVELS:
        raise HTTPException(422, f"level must be one of {LEVELS}")
    known = {v.key for v in VARIABLES[q.level]}
    unknown = [v for v in needed if v not in known]
    if unknown:
        raise HTTPException(422, f"Unknown variables for level {q.level}: {unknown}")
    df = frame(s, q.level, q.project_ids or None, max(1, q.abandon_months))
    if q.min_active_months > 1 and not df.empty:
        active = df.groupby(["project", "developer"])["developer"].transform("size") if q.level == "developer_month" else df["active_months"]
        df = df[active >= q.min_active_months]
    return df


@app.post("/api/analysis/correlations")
def correlations(q: CorrelationQuery):
    if q.method not in analysis.METHODS:
        raise HTTPException(422, f"method must be one of {analysis.METHODS}")
    if len([v for v in q.variables if v not in q.controls]) < 2:
        raise HTTPException(422, "Select at least two variables (not counting controls)")
    with session_scope() as s:
        df = analysis_frame(s, q, [*q.variables, *q.controls])
    result = analysis.correlation_matrix(df, q.variables, q.method, q.controls)
    warnings = []
    if q.level == "developer_month":
        warnings.append(
            "Developer-month rows are repeated measures of the same developers: p-values assume independent "
            "observations and are therefore too optimistic. Use them descriptively, or fit a mixed-effects model on the export."
        )
    projects = df["project"].nunique() if not df.empty else 0
    if projects > 1:
        warnings.append(
            f"Rows pooled from {projects} projects: correlations mix within- and between-project variation."
        )
    if len(df) < 30:
        warnings.append(f"Only {len(df)} rows: estimates are unstable.")
    return {**result, "rows": len(df), "developers": int(df[["project", "developer"]].drop_duplicates().shape[0]) if not df.empty else 0,
            "projects": int(projects), "method": q.method, "controls": q.controls, "level": q.level, "warnings": warnings}  # fmt: skip


@app.post("/api/analysis/scatter")
def scatter(q: ScatterQuery, limit: int = 3000):
    with session_scope() as s:
        df = analysis_frame(s, q, [q.x, q.y])
    d = df.dropna(subset=[q.x, q.y])
    total = len(d)
    if total > limit:
        d = d.sample(limit, random_state=0)
    label_cols = ["project", "developer"] + (["month"] if q.level == "developer_month" else [])
    points = [
        {"x": float(r[q.x]), "y": float(r[q.y]), "label": " · ".join(str(r[c]) for c in label_cols)}
        for _, r in d.iterrows()
    ]
    return {"points": points, "total": total, "sampled": total > limit}


# -- survival ---------------------------------------------------------------------


@app.get("/api/survival/options")
def survival_options():
    return {
        "covariates": [
            {"key": c.key, "label": c.label, "description": c.description, "default": c.default}
            for c in survival.COVARIATES
        ],
        "groupings": [{"key": k, "label": v} for k, v in survival.GROUPINGS.items()],
        "clusters": [{"key": k, "label": v} for k, v in survival.CLUSTERS.items()],
        "ph_transforms": list(survival.PH_TRANSFORMS),
        "defaults": {"abandon_months": settings.abandon_months, "window": 3, "min_active_months": 1},
    }


class SurvivalQuery(BaseModel):
    project_ids: list[int] | None = None
    abandon_months: int = settings.abandon_months
    window: int = 3
    min_active_months: int = 1
    covariates: list[str] = [c.key for c in survival.COVARIATES if c.default]
    log_counts: bool = True
    standardize: bool = True
    stratify: bool = True
    penalizer: float = 0.0
    cluster: str = "person"
    ph_transform: str = "rank"
    group: str = "core"


@app.post("/api/survival")
def survival_analysis(q: SurvivalQuery):
    unknown = [c for c in q.covariates if c not in survival.COVARIATE_KEYS]
    if unknown:
        raise HTTPException(422, f"Unknown covariates: {unknown}")
    if q.cluster not in survival.CLUSTERS:
        raise HTTPException(422, f"cluster must be one of {list(survival.CLUSTERS)}")
    if q.ph_transform not in survival.PH_TRANSFORMS:
        raise HTTPException(422, f"ph_transform must be one of {survival.PH_TRANSFORMS}")
    if q.group not in survival.GROUPINGS:
        raise HTTPException(422, f"group must be one of {list(survival.GROUPINGS)}")
    with session_scope() as s:
        intervals, stats = survival.build_intervals(
            s, q.project_ids or None, max(1, q.abandon_months), max(1, q.window), max(1, q.min_active_months)
        )
    out = {"dataset": {**stats, "at_risk_subjects": int(intervals["sid"].nunique()) if len(intervals) else 0,
                       "intervals": len(intervals)}}  # fmt: skip
    if intervals.empty or intervals["event"].sum() == 0:
        out["cox"] = {"error": "No abandonment events in this selection."}
        out["km"] = {"groups": [], "logrank_p": None}
        return out
    try:
        out["cox"] = (
            survival.fit_cox(
                intervals, q.covariates, q.log_counts, q.standardize, q.stratify, max(0.0, q.penalizer),
                q.cluster, q.ph_transform,
            )  # fmt: skip
            if q.covariates
            else {"error": "Select at least one covariate."}
        )
    except Exception as e:  # convergence failures are reported, not raised
        out["cox"] = {"error": f"Model did not converge: {e}. Try fewer covariates or a small penalizer (e.g. 0.1)."}
    out["km"] = survival.kaplan_meier(intervals, q.group)
    return out


# -- frontend (production build) -----------------------------------------------------

DIST = Path(BACKEND_DIR.parent / "frontend" / "dist")
if DIST.exists():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        file = DIST / path
        return FileResponse(file if path and file.is_file() else DIST / "index.html")
