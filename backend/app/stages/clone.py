import os
import shutil
import subprocess

from ..pipeline import Stage, StageContext


def git(*args: str, cwd=None) -> str:
    try:
        return subprocess.run(
            ["git", "-c", "core.quotePath=false", *args],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            # Never wait for credentials: a private or deleted repository must fail, not hang the worker.
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        ).stdout
    except subprocess.CalledProcessError as e:
        detail = (e.stderr or "").strip().splitlines()
        hint = " (deleted, private, or never existed)" if any("not found" in l.lower() for l in detail) else ""
        raise RuntimeError(f"git {args[0]} failed: {detail[-1] if detail else e}{hint}") from None


class CloneStage(Stage):
    name = "clone"
    label = "Clone"
    description = "Bare clone of the repository (fetch if already present)."

    def run(self, ctx: StageContext):
        path = ctx.project.repo_path
        url = f"https://github.com/{ctx.project.full_name}.git"
        if (path / "HEAD").exists():
            ctx.progress(0, 1, "Fetching updates")
            ctx.checkpoint()
            git("fetch", "--prune", "origin", "+refs/heads/*:refs/heads/*", cwd=path)
            ctx.log("Fetched existing clone")
        else:
            ctx.progress(0, 1, "Cloning")
            ctx.checkpoint()
            path.parent.mkdir(parents=True, exist_ok=True)
            # Partial leftovers from an interrupted clone would make git refuse the target.
            if path.exists():
                shutil.rmtree(path)
            git("clone", "--bare", "--quiet", url, str(path))
            ctx.log(f"Cloned {url}")
        count = int(git("rev-list", "--count", "HEAD", cwd=path).strip() or 0)
        ctx.progress(1, 1, f"{count} commits on default branch")
