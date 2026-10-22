import subprocess
from datetime import datetime, timezone

from sqlalchemy import select

from ..config import settings
from ..models import Commit, CommitFile
from ..pipeline import Stage, StageContext
from .clone import git

REC = "\x1e"
SEP = "\x1f"
FORMAT = f"--format={REC}%H{SEP}%P{SEP}%an{SEP}%ae{SEP}%aI{SEP}%cI"


def _unquote(path: str) -> str:
    if len(path) >= 2 and path[0] == path[-1] == '"':
        return path[1:-1].encode("latin-1", "backslashreplace").decode("unicode_escape").encode("latin-1").decode(
            "utf-8", "replace"
        )
    return path


def _to_utc(iso: str) -> tuple[datetime, int]:
    dt = datetime.fromisoformat(iso)
    offset = dt.utcoffset()
    offset_min = int(offset.total_seconds() // 60) if offset else 0
    return dt.astimezone(timezone.utc).replace(tzinfo=None), offset_min


def parse_log(lines):
    """Parse `git log --raw --numstat` output into commit dicts.

    Raw lines (":old new sha sha STATUS\tpath[\tnew]") and numstat lines ("add\tdel\tpath")
    come in the same file order, so they are paired by position.
    """
    current = None
    for line in lines:
        line = line.rstrip("\n")
        if line.startswith(REC):
            if current:
                yield current
            sha, parents, name, email, adate, cdate = line[1:].split(SEP)
            current = {
                "sha": sha,
                "parents": parents.split(),
                "name": name,
                "email": email,
                "authored": adate,
                "committed": cdate,
                "raw": [],
                "num": [],
            }
        elif not line or current is None:
            continue
        elif line.startswith(":"):
            meta, *paths = line.split("\t")
            status = meta.split()[-1][0]
            paths = [_unquote(p) for p in paths]
            if status in ("R", "C") and len(paths) == 2:
                current["raw"].append((status, paths[1], paths[0]))
            else:
                current["raw"].append((status, paths[0], None))
        else:
            add, dele, *_ = line.split("\t")
            current["num"].append((int(add) if add.isdigit() else 0, int(dele) if dele.isdigit() else 0))
    if current:
        yield current


class CommitsStage(Stage):
    name = "commits"
    label = "Commits"
    description = "Chronological git history: authors, time zones, touched files, renames, line counts."

    def run(self, ctx: StageContext):
        repo = ctx.project.repo_path
        total = int(git("rev-list", "--count", "HEAD", cwd=repo).strip() or 0)
        known = set(ctx.db.scalars(select(Commit.sha).where(Commit.project_id == ctx.project.id)))
        seq = ctx.cursor.get("seq", len(known))
        last = ctx.cursor.get("last_sha")
        rev = "HEAD"
        if last and subprocess.run(["git", "cat-file", "-e", last], cwd=repo).returncode == 0:
            rev = f"{last}..HEAD"
        ctx.progress(len(known), total, f"Resuming after {last[:8]}" if last else "Reading history")
        ctx.checkpoint()

        proc = subprocess.Popen(
            ["git", "-c", "core.quotePath=false", "log", rev, "--reverse", "--date-order", "-M",
             "--raw", "--numstat", "--no-abbrev", "--diff-merges=off", FORMAT],
            cwd=repo, stdout=subprocess.PIPE, text=True, encoding="utf-8", errors="replace",
        )  # fmt: skip
        pending = 0
        try:
            for c in parse_log(proc.stdout):
                if c["sha"] in known:
                    continue
                authored, tz = _to_utc(c["authored"])
                committed, _ = _to_utc(c["committed"])
                email = c["email"].strip().lower()
                alias_id = ctx.alias_id(
                    f"git:{email}|{c['name'].strip()}"[:600], "git", name=c["name"].strip(), email=email
                )
                nums = c["num"] + [(0, 0)] * (len(c["raw"]) - len(c["num"]))
                commit = Commit(
                    project_id=ctx.project.id, sha=c["sha"], seq=seq, alias_id=alias_id,
                    authored_at=authored, tz_offset_min=tz, committed_at=committed,
                    is_merge=len(c["parents"]) > 1,
                    additions=sum(a for a, _ in nums), deletions=sum(d for _, d in nums),
                    files_changed=len(c["raw"]),
                )  # fmt: skip
                ctx.db.add(commit)
                ctx.db.flush()
                ctx.db.add_all(
                    CommitFile(commit_id=commit.id, path=path, old_path=old, change_type=status if status in "AMDR" else "M",
                               additions=a, deletions=d)
                    for (status, path, old), (a, d) in zip(c["raw"], nums)
                )  # fmt: skip
                known.add(c["sha"])
                seq += 1
                pending += 1
                ctx.cursor.update(last_sha=c["sha"], seq=seq)
                if pending >= settings.batch_size:
                    ctx.progress(len(known), total, f"{len(known)}/{total} commits")
                    ctx.checkpoint()
                    pending = 0
            if proc.wait() != 0:
                raise RuntimeError(f"git log exited with {proc.returncode}")
        finally:
            if proc.poll() is None:
                proc.kill()
        ctx.progress(len(known), max(total, len(known)), f"{len(known)} commits")
        ctx.log(f"Mined {len(known)} commits")
