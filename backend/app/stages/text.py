"""Text classification stages: toxicity (ToxiCR) and sentiment polarity (Senti4SD).

Both tools run in their own Python environments under EXTERNAL_DIR, as persistent subprocesses, so their
pinned dependencies and licenses stay separate from the backend. No text leaves the machine.
"""

import json
import os
import re
import subprocess
from pathlib import Path

from sqlalchemy import select

from ..config import settings
from ..models import Alias, Comment, Issue
from ..pipeline import Stage, StageContext

TOOLS_DIR = Path(__file__).resolve().parent.parent / "tools"

_FENCE = re.compile(r"```.*?(```|$)", re.S)
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.S)
_QUOTE = re.compile(r"^\s*>.*$", re.M)
# GitHub @mentions are user names, not the author's words; ToxiCR's profanity tokenizer would split e.g.
# "@parikshit-adhikari" into "parik shit adhikari" and flag it.
_MENTION = re.compile(r"(?<![\w.])@[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})(?:/[A-Za-z0-9_.-]+)?")
_WS = re.compile(r"\s+")


def clean_text(text: str | None) -> str:
    """Keep only what the author wrote: drop quoted replies (`> ...`), fenced code, HTML comments (issue
    templates) and @mentions. A quoted toxic sentence would otherwise be attributed to the person replying."""
    if not text:
        return ""
    text = _HTML_COMMENT.sub(" ", text)
    text = _FENCE.sub(" ", text)
    text = _QUOTE.sub(" ", text)
    text = _MENTION.sub(" ", text)
    return _WS.sub(" ", text).strip()[:5000]


class ExternalClassifier:
    """A long-running classifier process speaking JSON lines (see app/tools/*_runner.py)."""

    def __init__(self, python: Path, runner: str, *args: str, env: dict | None = None):
        self.proc = subprocess.Popen(
            [str(python), str(TOOLS_DIR / runner), *args],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env={**os.environ, **(env or {})},
        )
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError(f"{runner} failed to start: {self.proc.stderr.read()[-1500:]}")
            if line.strip() == "READY":
                break

    def classify(self, batch: list[dict]) -> list[dict]:
        self.proc.stdin.write(json.dumps(batch) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError(f"classifier crashed: {self.proc.stderr.read()[-1500:]}")
        return json.loads(line)

    def close(self):
        if self.proc.poll() is None:
            self.proc.stdin.close()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()


class TextStage(Stage):
    """Classify every issue (title + body), comment and review body that has no score yet."""

    column = ""  # Issue/Comment column holding the result
    empty_value: object = None
    batch_size = 256

    def tool_status(self) -> str | None:
        """None when the tool is installed, otherwise a message explaining what is missing."""
        raise NotImplementedError

    def open(self) -> ExternalClassifier:
        raise NotImplementedError

    def store(self, row, result: dict):
        raise NotImplementedError

    def run(self, ctx: StageContext):
        missing = self.tool_status()
        if missing:
            ctx.cursor["skipped"] = True  # re-run automatically once the tool is available (worker start-up)
            ctx.progress(0, 0, f"Skipped: {missing}")
            ctx.log(f"{self.label} skipped: {missing}", "warn")
            return
        ctx.cursor.pop("skipped", None)
        pid = ctx.project.id
        bots = set(ctx.db.scalars(select(Alias.id).where(Alias.project_id == pid, Alias.is_bot)))
        col_i, col_c = getattr(Issue, self.column), getattr(Comment, self.column)
        todo_i = list(ctx.db.scalars(select(Issue).where(Issue.project_id == pid, col_i.is_(None))))
        todo_c = list(ctx.db.scalars(select(Comment).where(Comment.project_id == pid, col_c.is_(None))))
        rows = [(r, f"{r.title or ''}. {r.body or ''}") for r in todo_i] + [(r, r.body) for r in todo_c]
        total, done = len(rows), 0
        work = []
        for row, raw in rows:
            text = clean_text(raw)
            if row.alias_id in bots or len(text) < 3:
                setattr(row, self.column, self.empty_value)  # nothing (human) to classify
                done += 1
            else:
                work.append((row, text))
        ctx.progress(done, total, f"{len(work)} texts to classify")
        ctx.checkpoint()
        if not work:
            return
        clf = self.open()
        try:
            for start in range(0, len(work), self.batch_size):
                chunk = work[start : start + self.batch_size]
                results = clf.classify([{"id": i, "text": t} for i, (_, t) in enumerate(chunk)])
                for (row, _), res in zip(chunk, results):
                    self.store(row, res)
                done += len(chunk)
                ctx.progress(done, total, f"{done}/{total} texts")
                ctx.checkpoint()
        finally:
            clf.close()
        ctx.log(f"Classified {len(work)} texts ({total - len(work)} skipped: bots or empty)")


class ToxicityStage(TextStage):
    name = "toxicity"
    label = "Toxicity"
    description = "ToxiCR (Sarker et al., TOSEM 2023): random forest on TF-IDF, run locally."
    column = "toxic"
    empty_value = -1

    @staticmethod
    def paths():
        base = settings.external_dir
        model = base / "ToxiCR" / "pre-trained" / "model-RF-tfidf-profane-False-keyword-True-split-False.pickle"
        return base / "toxicr-venv" / "bin" / "python", base / "ToxiCR", model, base / "nltk_data"

    def tool_status(self):
        if not settings.enable_toxicity:
            return "disabled: ToxiCR is not validated on GitHub comments yet (set ENABLE_TOXICITY=true to run it)"
        python, repo, model, _ = self.paths()
        if not python.exists() or not repo.exists():
            return "ToxiCR is not installed (see README, Text classification)"
        if not model.exists():
            return "ToxiCR model not trained yet (run its --mode pretrain, see README)"
        return None

    def open(self):
        python, repo, _, nltk = self.paths()
        return ExternalClassifier(python, "toxicr_runner.py", str(repo), env={"NLTK_DATA": str(nltk)})

    def store(self, row, result):
        row.toxic = result["toxic"]
        row.toxicity_p = result["p"]


class SentimentStage(TextStage):
    name = "sentiment"
    label = "Sentiment"
    description = "Senti4SD (Calefato et al., EMSE 2018) retrained on GitHub (Novielli et al., MSR 2020), run locally."
    column = "sentiment"
    empty_value = "none"
    batch_size = 2000  # each batch starts the Java feature extractor, which loads 835 MB of embeddings

    @staticmethod
    def paths():
        base = settings.external_dir
        return base / "senti4sd-venv" / "bin" / "python", base / "pySenti4SD"

    def tool_status(self):
        python, repo = self.paths()
        needed = [repo / "java" / "Senti4SD-fast.jar", repo / "java" / "dsm.bin", repo / "Senti4SD-github.model"]
        if not python.exists() or not all(f.exists() for f in needed):
            return "Senti4SD is not installed (see README, Text classification)"
        return None

    def open(self):
        python, repo = self.paths()
        return ExternalClassifier(python, "senti4sd_runner.py", str(repo))

    def store(self, row, result):
        row.sentiment = result["polarity"]


_AT_MENTION = re.compile(r"(?<![\w.])@[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})")


def conversation_text(issue: Issue, comments: list[Comment], bots: set[int]) -> str:
    """Issue/PR body then conversation comments (chronological), human authors only, raw text, speakers as
    P1, P2, ... and @mentions as @user; empty if no human wrote anything."""
    turns = [(issue.alias_id, issue.body)] + [(c.alias_id, c.body) for c in comments]
    speakers: dict[int, str] = {}
    lines = []
    for alias, body in turns:
        if alias is None or alias in bots or not (body or "").strip():
            continue
        speakers.setdefault(alias, f"P{len(speakers) + 1}")
        lines.append(f"{speakers[alias]}: {_AT_MENTION.sub('@user', body)}")
    return "\n".join(lines)


class ConversationToxicityStage(Stage):
    """Toxicity of whole conversations, the unit the classifier was trained and validated on (docs/TOXICITY.md).

    A conversation is rebuilt like the training data: the issue/PR body followed by the conversation comments
    (not inline review comments) in chronological order, human authors only, raw text, speakers as P1, P2, ...
    and @mentions as @user.
    """

    name = "conversation_toxicity"
    label = "Conversation toxicity"
    description = "RoBERTa fine-tuned on 124k GitHub conversations (ASE 2026 data), validated at the natural toxicity rate."
    batch_size = 64

    @staticmethod
    def paths():
        base = settings.external_dir
        return base / "toxbert-venv" / "bin" / "python", base / "toxicity-gh" / "work" / "models" / "toxic-roberta-github"

    def tool_status(self):
        python, model = self.paths()
        if not python.exists() or not (model / "threshold.json").exists():
            return "conversation toxicity model not installed (see docs/TOXICITY.md)"
        return None

    def run(self, ctx: StageContext):
        missing = self.tool_status()
        if missing:
            ctx.cursor["skipped"] = True
            ctx.progress(0, 0, f"Skipped: {missing}")
            ctx.log(f"{self.label} skipped: {missing}", "warn")
            return
        ctx.cursor.pop("skipped", None)
        pid = ctx.project.id
        bots = set(ctx.db.scalars(select(Alias.id).where(Alias.project_id == pid, Alias.is_bot)))
        issues = list(ctx.db.scalars(select(Issue).where(Issue.project_id == pid, Issue.conv_toxic.is_(None))))
        comments: dict[int, list] = {}
        for c in ctx.db.scalars(
            select(Comment).where(Comment.project_id == pid, Comment.kind == "issue_comment").order_by(Comment.created_at)
        ):
            comments.setdefault(c.issue_number, []).append(c)
        work = []
        for issue in issues:
            text = conversation_text(issue, comments.get(issue.number, []), bots)
            if text:
                work.append((issue, text))
            else:
                issue.conv_toxic = -1
        ctx.progress(len(issues) - len(work), len(issues), f"{len(work)} conversations to classify")
        ctx.checkpoint()
        if not work:
            return
        python, model = self.paths()
        clf = ExternalClassifier(python, "convtox_runner.py", str(model), str(Path(__file__).resolve().parents[2] / "scripts" / "toxicity"))
        done = len(issues) - len(work)
        try:
            for start in range(0, len(work), self.batch_size):
                chunk = work[start : start + self.batch_size]
                for (issue, _), res in zip(chunk, clf.classify([{"id": i, "text": t} for i, (_, t) in enumerate(chunk)])):
                    issue.conv_toxic, issue.conv_toxicity_score = res["toxic"], res["p"]
                done += len(chunk)
                ctx.progress(done, len(issues), f"{done}/{len(issues)} conversations")
                ctx.checkpoint()
        finally:
            clf.close()
        toxic = sum(i.conv_toxic == 1 for i, _ in work)
        ctx.log(f"Classified {len(work)} conversations: {toxic} toxic ({toxic / len(work):.2%})")
