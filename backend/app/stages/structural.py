"""Truck factor (Avelino et al., ICPC 2016) and core/peripheral classification (Joblin et al., ICSE 2017)."""

import math
from collections import Counter, defaultdict
from itertools import combinations

import networkx as nx
from sqlalchemy import delete, select, update

from ..config import settings
from ..models import Alias, Comment, Commit, CommitFile, Developer, DeveloperMonth, Issue, ProjectMetric
from ..pipeline import Stage, StageContext
from .aggregate import month_of
from .clone import git

DOA_THRESHOLD = 3.293
NORMALIZED_DOA_THRESHOLD = 0.75


def doa(first_author: bool, deliveries: int, others: int) -> float:
    return 3.293 + 1.098 * first_author + 0.164 * deliveries - 0.321 * math.log(1 + others)


def file_authors(history: dict[str, dict]) -> dict[str, set[int]]:
    """Map each file to its authors according to normalized DOA."""
    authors: dict[str, set[int]] = {}
    for path, h in history.items():
        changes: Counter = h["changes"]
        total = sum(changes.values())
        scores = {dev: doa(dev == h["creator"], n, total - n) for dev, n in changes.items()}
        if not scores:
            authors[path] = set()
            continue
        top = max(scores.values())
        authors[path] = {
            dev for dev, s in scores.items() if s >= DOA_THRESHOLD and top > 0 and s / top > NORMALIZED_DOA_THRESHOLD
        }
    return authors


def truck_factor(authors: dict[str, set[int]]) -> tuple[int, list[int], dict[int, int]]:
    """Greedily remove top authors until more than 50% of files have no author left."""
    files_of: dict[int, set[str]] = defaultdict(set)
    for path, devs in authors.items():
        for d in devs:
            files_of[d].add(path)
    authored_count = {d: len(f) for d, f in files_of.items()}
    total = len(authors)
    if total == 0:
        return 0, [], authored_count
    remaining = {p: set(d) for p, d in authors.items()}
    removed: list[int] = []
    while sum(1 for d in remaining.values() if not d) / total <= 0.5 and files_of:
        top = max(files_of, key=lambda d: (len(files_of[d]), -d))
        removed.append(top)
        for p in files_of.pop(top):
            remaining[p].discard(top)
    return len(removed), removed, authored_count


def truck_factor_history(db, pid: int, human) -> dict[str, dict]:
    """Truck factor as it stood at the end of each month, from the commits made up to then only, so it can enter
    a model at time t without using the future. Files are those alive at that point of the replay (deletions
    and renames followed), not HEAD's. Returns {"YYYY-MM": {"tf": n, "members": [developer ids]}} for every
    month in which the history changed; between keys the previous value holds."""
    history: dict[str, dict] = {}
    out: dict[str, dict] = {}
    current = None  # latest month seen: commit dates are not monotonic in log order

    def snapshot(month):
        tf, members, _ = truck_factor(file_authors(history))
        out[month] = {"tf": tf, "members": members}

    q = (
        select(Commit.alias_id, Commit.authored_at, CommitFile.path, CommitFile.old_path, CommitFile.change_type)
        .join(CommitFile, CommitFile.commit_id == Commit.id)
        .where(Commit.project_id == pid)
        .order_by(Commit.seq, CommitFile.id)
    )
    for alias_id, authored, path, old_path, change in db.execute(q):
        month = month_of(authored)
        if current is not None and month > current:
            snapshot(current)
        current = month if current is None else max(current, month)
        dev = human(alias_id)
        if change == "D":
            history.pop(path, None)
            continue
        if change == "R" and old_path:
            h = history.pop(old_path, None) or {"creator": None, "changes": Counter()}
        else:
            h = history.get(path) or {"creator": dev if change == "A" else None, "changes": Counter()}
        history[path] = h
        if dev is not None:
            h["changes"][dev] += 1
    if current is not None:
        snapshot(current)
    return out


def month_index(month: str) -> int:
    y, m = month.split("-")
    return int(y) * 12 + int(m) - 1


class StructuralStage(Stage):
    name = "structural"
    label = "Structural metrics"
    description = "Truck factor (DOA) and core/peripheral roles from developer-network centrality."

    def run(self, ctx: StageContext):
        pid = ctx.project.id
        dev_of = dict(ctx.db.execute(select(Alias.id, Alias.developer_id).where(Alias.project_id == pid)).all())
        bots = set(ctx.db.scalars(select(Developer.id).where(Developer.project_id == pid, Developer.is_bot)))

        def human(alias_id):
            d = dev_of.get(alias_id)
            return None if d is None or d in bots else d

        # -- replay file history -----------------------------------------------------
        ctx.progress(0, 2, "Replaying file history")
        ctx.checkpoint()
        history: dict[str, dict] = {}
        files_by_month: dict[str, dict[str, set[int]]] = defaultdict(lambda: defaultdict(set))
        q = (
            select(Commit.alias_id, Commit.authored_at, CommitFile.path, CommitFile.old_path, CommitFile.change_type)
            .join(CommitFile, CommitFile.commit_id == Commit.id)
            .where(Commit.project_id == pid)
            .order_by(Commit.seq, CommitFile.id)
        )
        for alias_id, authored, path, old_path, change in ctx.db.execute(q):
            dev = human(alias_id)
            if change == "D":
                history.pop(path, None)
                continue
            if change == "R" and old_path:
                h = history.pop(old_path, None) or {"creator": None, "changes": Counter()}
            else:
                h = history.get(path) or {"creator": dev if change == "A" else None, "changes": Counter()}
            history[path] = h
            if dev is not None:
                h["changes"][dev] += 1
                files_by_month[month_of(authored)][path].add(dev)

        head_files = set(git("ls-tree", "-r", "--name-only", "HEAD", cwd=ctx.project.repo_path).splitlines())
        authors = file_authors({p: h for p, h in history.items() if p in head_files})
        tf, members, authored_count = truck_factor(authors)

        ctx.db.execute(
            update(Developer).where(Developer.project_id == pid).values(truck_factor_member=False, files_authored=0)
        )
        for dev in ctx.db.scalars(select(Developer).where(Developer.project_id == pid)):
            dev.files_authored = authored_count.get(dev.id, 0)
            dev.truck_factor_member = dev.id in members
        ctx.db.execute(delete(ProjectMetric).where(ProjectMetric.project_id == pid, ProjectMetric.key == "truck_factor"))
        ctx.db.add(
            ProjectMetric(
                project_id=pid,
                key="truck_factor",
                value={
                    "value": tf,
                    "members": members,
                    "files": len(authors),
                    "files_with_author": sum(1 for d in authors.values() if d),
                },
            )
        )
        ctx.log(f"Truck factor = {tf} over {len(authors)} files")
        ctx.db.execute(
            delete(ProjectMetric).where(ProjectMetric.project_id == pid, ProjectMetric.key == "truck_factor_history")
        )
        ctx.db.add(ProjectMetric(project_id=pid, key="truck_factor_history", value=truck_factor_history(ctx.db, pid, human)))
        ctx.progress(1, 2, f"Truck factor {tf}; building developer networks")
        ctx.checkpoint()

        # -- developer network per sliding window ------------------------------------
        threads_by_month: dict[str, dict[int, set[int]]] = defaultdict(lambda: defaultdict(set))
        for alias_id, created, number in ctx.db.execute(
            select(Issue.alias_id, Issue.created_at, Issue.number).where(Issue.project_id == pid)
        ):
            if (dev := human(alias_id)) is not None:
                threads_by_month[month_of(created)][number].add(dev)
        for alias_id, created, number in ctx.db.execute(
            select(Comment.alias_id, Comment.created_at, Comment.issue_number).where(Comment.project_id == pid)
        ):
            if (dev := human(alias_id)) is not None:
                threads_by_month[month_of(created)][number].add(dev)

        dm_rows = list(ctx.db.scalars(select(DeveloperMonth).where(DeveloperMonth.project_id == pid)))
        by_month: dict[str, list[DeveloperMonth]] = defaultdict(list)
        for r in dm_rows:
            r.centrality, r.degree, r.is_core = 0.0, 0, False
            by_month[r.month].append(r)

        window = settings.network_window_months
        months_with_data = sorted(set(files_by_month) | set(threads_by_month))
        idx = {m: month_index(m) for m in months_with_data}
        for n, month in enumerate(sorted(by_month), 1):
            mi = month_index(month)
            in_window = [m for m in months_with_data if mi - window < idx[m] <= mi]
            g = nx.Graph()
            for groups in (files_by_month, threads_by_month):
                merged: dict = defaultdict(set)
                for m in in_window:
                    for key, devs in groups[m].items():
                        merged[key] |= devs
                for devs in merged.values():
                    g.add_nodes_from(devs)
                    for a, b in combinations(sorted(devs), 2):
                        w = g.get_edge_data(a, b, {}).get("weight", 0)
                        g.add_edge(a, b, weight=w + 1)
            centrality = self._centrality(g)
            core = self._core(centrality)
            for r in by_month[month]:
                r.centrality = centrality.get(r.developer_id, 0.0)
                r.degree = g.degree(r.developer_id) if r.developer_id in g else 0
                r.is_core = r.developer_id in core
            if n % 12 == 0:
                ctx.progress(1, 2, f"Networks: {n}/{len(by_month)} months")
        ctx.progress(2, 2, f"Truck factor {tf}; networks for {len(by_month)} months")
        ctx.log(f"Computed developer networks for {len(by_month)} months (window {window} months)")

    @staticmethod
    def _centrality(g: nx.Graph) -> dict[int, float]:
        if g.number_of_nodes() == 0:
            return {}
        if g.number_of_edges() == 0:
            return {n: 0.0 for n in g}
        try:
            return nx.eigenvector_centrality(g, max_iter=2000, tol=1e-6, weight="weight")
        except nx.PowerIterationFailedConvergence:
            return nx.degree_centrality(g)

    @staticmethod
    def _core(centrality: dict[int, float]) -> set[int]:
        """Developers that together account for CORE_THRESHOLD of cumulative centrality."""
        total = sum(centrality.values())
        if total <= 0:
            return set()
        core, acc = set(), 0.0
        for dev, c in sorted(centrality.items(), key=lambda kv: -kv[1]):
            if acc >= settings.core_threshold * total:
                break
            core.add(dev)
            acc += c
        return core
