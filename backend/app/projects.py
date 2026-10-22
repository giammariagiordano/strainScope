"""Project deletion shared by the API and the worker."""

import shutil
from datetime import timedelta

from sqlalchemy import delete, select

from .config import settings
from .db import session_scope
from .models import Project, utcnow
from .pipeline import ProjectRef


def purge(pid: int) -> bool:
    """Delete a project, all its rows (FK cascade) and its clone. Returns False if it no longer exists."""
    with session_scope() as s:
        p = s.get(Project, pid)
        if p is None:
            return False
        ref = ProjectRef(p.id, p.owner, p.name)
        s.execute(delete(Project).where(Project.id == pid))
    shutil.rmtree(ref.repo_path, ignore_errors=True)
    return True


def worker_alive_for(p: Project) -> bool:
    cutoff = utcnow() - timedelta(seconds=settings.heartbeat_timeout_s)
    return p.heartbeat_at is not None and p.heartbeat_at >= cutoff


def request_delete(pid: int) -> str:
    """Delete now, or ask the worker to delete at its next checkpoint if it is mining this project.

    Returns "deleted", "deleting" or "missing".
    """
    with session_scope() as s:
        p = s.get(Project, pid)
        if p is None:
            return "missing"
        if p.status in ("running", "deleting") and worker_alive_for(p):
            p.status = "deleting"
            p.pause_requested = True  # makes the worker stop at its next checkpoint
            return "deleting"
    purge(pid)
    return "deleted"


def all_project_ids() -> list[int]:
    with session_scope() as s:
        return list(s.scalars(select(Project.id)))
