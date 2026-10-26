"""Pipeline worker: claims queued projects and runs their stages in order.

Run with `python -m app.worker`. Several workers can run against PostgreSQL;
use a single worker with SQLite.
"""

import logging
import os
import signal
import traceback
from datetime import timedelta

from sqlalchemy import select, update
from sqlalchemy.exc import OperationalError

from .config import settings
from .db import SessionLocal, init_db, session_scope
from .models import AppState, Log, Project, StageRun, utcnow
from .projects import purge
from .pipeline import STOP, Interrupted, Paused, ProjectRef, StageContext, ensure_stage_runs, registry

log = logging.getLogger("worker")


def _handle_signal(*_):
    STOP.set()
    log.info("stop requested: saving at the next checkpoint (press Ctrl+C again to force)")
    signal.signal(signal.SIGINT, signal.SIG_DFL)


def recover_orphans():
    """Projects left `running` by a dead worker go back to the queue; pending deletions are completed."""
    cutoff = utcnow() - timedelta(seconds=settings.heartbeat_timeout_s)
    with session_scope() as s:
        pending = list(s.scalars(select(Project.id).where(Project.status == "deleting")))
    for pid in pending:
        purge(pid)
        log.info("completed pending deletion of project %s", pid)
    with session_scope() as s:
        stale = list(
            s.scalars(
                select(Project).where(
                    Project.status == "running",
                    (Project.heartbeat_at.is_(None)) | (Project.heartbeat_at < cutoff),
                )
            )
        )
        for p in stale:
            p.status = "paused" if p.pause_requested else "queued"
            s.execute(
                update(StageRun)
                .where(StageRun.project_id == p.id, StageRun.status == "running")
                .values(status="pending")
            )
            s.add(Log(project_id=p.id, level="warn", message="Recovered after worker interruption"))
            log.info("recovered orphaned project %s", p.full_name)


def upgrade_pipelines():
    """Add stages introduced by a newer version to existing projects and recompute what depends on them.

    Mining stages already done are kept; finished projects go back to the queue only to run the new
    stages and the ones after them (e.g. text classification, then aggregation).
    """
    stages = {st.name: st for st in registry() if hasattr(st, "tool_status")}
    with session_scope() as s:
        for p in s.scalars(select(Project).where(Project.status != "deleting")):
            ensure_stage_runs(s, p.id)
            if p.status != "done":
                continue
            s.flush()
            runs = list(s.scalars(select(StageRun).where(StageRun.project_id == p.id).order_by(StageRun.position)))
            # A stage skipped because its tool was missing runs again once the tool is installed.
            for r in runs:
                stage = stages.get(r.stage)
                skipped = (r.cursor or {}).get("skipped") or (r.message or "").startswith("Skipped")  # older runs
                if r.status == "done" and skipped and stage and stage.tool_status() is None:
                    r.status, r.cursor = "pending", {}
            first = next((r.position for r in runs if r.status != "done"), None)
            if first is None:
                continue
            # Stages after one that never ran depend on it: recompute them too.
            for r in runs:
                if r.position > first and r.status == "done":
                    r.status, r.cursor, r.done, r.total, r.message = "pending", {}, 0, None, None
            p.status, p.finished_at = "queued", None
            s.add(Log(project_id=p.id, message="New pipeline stages added: computing them on the mined data"))
            log.info("upgraded pipeline of %s", p.full_name)


def claim() -> ProjectRef | None:
    with session_scope() as s:
        candidates = s.scalars(
            select(Project.id).where(Project.status == "queued").order_by(Project.created_at).limit(5)
        ).all()
        for pid in candidates:
            # Atomic claim: only one worker can flip queued -> running.
            res = s.execute(
                update(Project)
                .where(Project.id == pid, Project.status == "queued")
                .values(status="running", heartbeat_at=utcnow(), error=None)
            )
            if res.rowcount == 1:
                p = s.get(Project, pid)
                ensure_stage_runs(s, pid)
                return ProjectRef(p.id, p.owner, p.name, p.since)
    return None


def heartbeat():
    with session_scope() as s:
        s.merge(AppState(key="worker", value={"alive_at": utcnow().isoformat(), "pid": os.getpid()}))


def run_project(ref: ProjectRef):
    log.info("running %s", ref.full_name)
    for stage in registry():
        if STOP.is_set():
            _set_project(ref.id, "queued")
            return
        db = SessionLocal()
        ctx = None
        try:
            run = db.scalar(select(StageRun).where(StageRun.project_id == ref.id, StageRun.stage == stage.name))
            if run.status == "done":
                continue
            run.status = "running"
            run.started_at = run.started_at or utcnow()
            run.message = None
            project = db.get(Project, ref.id)
            project.current_stage = stage.name
            db.commit()

            ctx = StageContext(db, ref, run)
            ctx.check_pause()
            stage.run(ctx)

            run.status = "done"
            run.finished_at = utcnow()
            run.cursor = dict(ctx.cursor)
            db.commit()
            heartbeat()
        except Interrupted:
            db.rollback()
            _mark_stage(ref.id, stage.name, "pending")
            _set_project(ref.id, "queued", log_message=f"Worker stopped during {stage.label}; will resume")
            return
        except Paused:
            db.rollback()  # discard the unfinished batch; the cursor points before it
            _mark_stage(ref.id, stage.name, "paused")
            _set_project(ref.id, "paused", log_message=f"Paused during {stage.label}")
            return
        except Exception as e:
            db.rollback()
            tb = traceback.format_exc()
            log.error("stage %s failed for %s\n%s", stage.name, ref.full_name, tb)
            _mark_stage(ref.id, stage.name, "failed", message=str(e)[:500])
            _set_project(ref.id, "failed", error=f"{stage.label}: {e}", log_message=tb[-4000:], level="error")
            return
        finally:
            if ctx:
                ctx.close()
            db.close()
    _set_project(ref.id, "done", log_message="Pipeline completed")


def _mark_stage(pid: int, stage: str, status: str, message: str | None = None):
    with session_scope() as s:
        s.execute(
            update(StageRun)
            .where(StageRun.project_id == pid, StageRun.stage == stage)
            .values(status=status, **({"message": message} if message else {}))
        )


def _set_project(pid: int, status: str, error: str | None = None, log_message: str | None = None, level="info"):
    with session_scope() as s:
        p = s.get(Project, pid)
        if p is None:
            return
        deleting = p.status == "deleting"
    if deleting:  # the user deleted it while it was being mined
        purge(pid)
        log.info("deleted project %s on request", pid)
        return
    with session_scope() as s:
        p = s.get(Project, pid)
        if p is None:
            return
        p.status = status
        p.error = error
        p.current_stage = None if status == "done" else p.current_stage
        if status in ("done", "paused"):
            p.pause_requested = False
        if status == "done":
            p.finished_at = utcnow()
        if log_message:
            s.add(Log(project_id=pid, level=level, message=log_message))


def _retry(fn, attempts: int = 10):
    for i in range(attempts):
        try:
            return fn()
        except OperationalError as e:
            if i == attempts - 1:
                raise
            log.warning("database busy (%s), retrying", e.orig)
            STOP.wait(2)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    _retry(init_db)
    _retry(recover_orphans)
    _retry(upgrade_pipelines)
    if not settings.github_tokens:
        log.warning("GITHUB_TOKENS not set: GitHub API limited to 60 requests/hour")
    log.info("worker started (db=%s)", settings.database_url.split("@")[-1])
    while not STOP.is_set():
        try:
            heartbeat()
            ref = claim()
        except OperationalError as e:  # e.g. SQLite busy: try again shortly
            log.warning("database busy: %s", e.orig)
            STOP.wait(2)
            continue
        if ref is None:
            STOP.wait(2)
            continue
        run_project(ref)
    log.info("worker stopped")


if __name__ == "__main__":
    main()
