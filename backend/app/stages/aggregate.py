from collections import defaultdict
from datetime import timedelta

from sqlalchemy import delete, select

from ..config import settings
from ..models import Alias, Comment, Commit, Developer, DeveloperMonth, Issue
from ..pipeline import Stage, StageContext
from ..psych import grateful, pronouns
from .text import clean_text


EMOTIONS = ("anger", "joy", "love", "sadness", "fear", "surprise")
EMOTION_COLUMNS = ("emotion_messages", *(f"{e}_emitted" for e in EMOTIONS), "emotion_received", "anger_received")


def month_of(dt) -> str:
    return f"{dt.year:04d}-{dt.month:02d}"


def is_off_hours(local_dt) -> tuple[bool, bool]:
    """Return (off_hours, weekend) for a commit in the author's local time."""
    start, end = settings.work_hours
    weekend = local_dt.weekday() >= 5
    return weekend or not (start <= local_dt.hour < end), weekend


class AggregateStage(Stage):
    name = "aggregate"
    label = "Monthly aggregation"
    description = "Developer × month activity: commits, lines, off-hours share, issues, PRs, comments, reviews."

    def run(self, ctx: StageContext):
        pid = ctx.project.id
        dev_of = dict(ctx.db.execute(select(Alias.id, Alias.developer_id).where(Alias.project_id == pid)).all())
        rows: dict[tuple[int, str], dict] = defaultdict(lambda: defaultdict(int))
        first: dict[int, object] = {}
        last: dict[int, object] = {}

        def touch(dev_id, when):
            if dev_id is None or when is None:
                return None
            if dev_id not in first or when < first[dev_id]:
                first[dev_id] = when
            if dev_id not in last or when > last[dev_id]:
                last[dev_id] = when
            return rows[(dev_id, month_of(when))]

        commits = ctx.db.execute(
            select(Commit.alias_id, Commit.authored_at, Commit.tz_offset_min, Commit.additions, Commit.deletions)
            .where(Commit.project_id == pid)
        )  # fmt: skip
        for alias_id, authored, tz, add, dele in commits:
            r = touch(dev_of.get(alias_id), authored)
            if r is None:
                continue
            off, weekend = is_off_hours(authored + timedelta(minutes=tz))
            r["commits"] += 1
            r["additions"] += add
            r["deletions"] += dele
            r["off_hours_commits"] += off
            r["weekend_commits"] += weekend

        bots = set(ctx.db.scalars(select(Developer.id).where(Developer.project_id == pid, Developer.is_bot)))
        has_toxicity = has_sentiment = has_reactions = has_emotions = False

        def feel(r, emotions):
            """Emotions (EmoTxt) of a classified message written by the developer."""
            nonlocal has_emotions
            if emotions is not None and emotions != "-":
                has_emotions = True
                r["emotion_messages"] += 1
                for e in filter(None, emotions.split(",")):
                    r[f"{e}_emitted"] += 1

        def psych(r, raw, pos, neg):
            """Psycholinguistic markers of a message written by the developer, and reactions it received."""
            nonlocal has_reactions
            text = clean_text(raw)
            words, we, i = pronouns(text)
            if words:
                r["psych_messages"] += 1
                r["words"] += words
                r["we_words"] += we
                r["i_words"] += i
                r["thanks_emitted"] += grateful(text)
            if pos is not None:
                has_reactions = True
                r["reacted_items"] += 1
                r["reactions_pos_received"] += pos
                r["reactions_neg_received"] += neg

        def emit(r, toxic, sentiment):
            """Count a classified message written by the developer."""
            nonlocal has_toxicity, has_sentiment
            if toxic in (0, 1):
                has_toxicity = True
                r["messages"] += 1
                r["toxic_emitted"] += toxic
            if sentiment in ("positive", "neutral", "negative"):
                has_sentiment = True
                r["sentiment_messages"] += 1
                r["negative_emitted"] += sentiment == "negative"

        issue_author: dict[int, int] = {}
        conv_toxic: dict[int, int | None] = {}
        participation: set[tuple[int, str, int]] = set()  # (developer, month, conversation)
        for number, alias_id, created, is_pr, toxic, sentiment, ctoxic, body, pos, neg, emotions in ctx.db.execute(
            select(Issue.number, Issue.alias_id, Issue.created_at, Issue.is_pr, Issue.toxic, Issue.sentiment, Issue.conv_toxic,
                   Issue.body, Issue.reactions_pos, Issue.reactions_neg, Issue.emotions)
            .where(Issue.project_id == pid)
        ):  # fmt: skip
            dev = dev_of.get(alias_id)
            conv_toxic[number] = ctoxic
            if dev is not None:
                issue_author[number] = dev
            if (r := touch(dev, created)) is not None:
                r["prs_opened" if is_pr else "issues_opened"] += 1
                emit(r, toxic, sentiment)
                psych(r, body, pos, neg)
                feel(r, emotions)
                participation.add((dev, month_of(created), number))

        received = []  # (author of the thread, month, toxic, sentiment, grateful) for messages written by others
        for number, alias_id, created, kind, toxic, sentiment, body, pos, neg, emotions in ctx.db.execute(
            select(Comment.issue_number, Comment.alias_id, Comment.created_at, Comment.kind, Comment.toxic, Comment.sentiment,
                   Comment.body, Comment.reactions_pos, Comment.reactions_neg, Comment.emotions)
            .where(Comment.project_id == pid)
        ):  # fmt: skip
            dev = dev_of.get(alias_id)
            if (r := touch(dev, created)) is not None:
                r["reviews" if kind == "review" else "comments"] += 1
                emit(r, toxic, sentiment)
                psych(r, body, pos, neg)
                feel(r, emotions)
                participation.add((dev, month_of(created), number))
            owner = issue_author.get(number)
            if owner is not None and owner != dev and created is not None:
                # Bots thank automatically ("Thanks for your contribution!"): not recognition by people.
                thanks = None if dev is None or dev in bots else grateful(clean_text(body))
                received.append((owner, month_of(created), toxic, sentiment, thanks, emotions))
        for owner, month, toxic, sentiment, thanks, emotions in received:
            r = rows.get((owner, month))
            if r is None:
                continue  # receiving alone does not make a month active
            if thanks is not None:
                r["replies_received"] += 1
                r["thanks_received"] += thanks
                if emotions is not None and emotions != "-":
                    r["emotion_received"] += 1
                    r["anger_received"] += "anger" in emotions.split(",")
            if toxic in (0, 1):
                r["received"] += 1
                r["toxic_received"] += toxic
            if sentiment in ("positive", "neutral", "negative"):
                r["sentiment_received"] += 1
                r["negative_received"] += sentiment == "negative"

        # Exposure to toxic conversations: conversations the developer wrote in, and how many of them are toxic.
        has_conv = any(v in (0, 1) for v in conv_toxic.values())
        for dev, month, number in participation:
            r = rows.get((dev, month))
            if r is not None and conv_toxic.get(number) in (0, 1):
                r["threads"] += 1
                r["toxic_threads"] += conv_toxic[number]

        strain_cols = ("messages", "toxic_emitted", "received", "toxic_received", "negative_emitted", "negative_received")
        helper_cols = ("sentiment_messages", "sentiment_received")

        def month_row(dev_id, month, values):
            values = dict(values)
            # Both tools classify the same texts (same cleaning and bot rules), so one denominator serves both.
            if not has_toxicity:
                values["messages"] = values.get("sentiment_messages", 0)
                values["received"] = values.get("sentiment_received", 0)
            for c in helper_cols:
                values.pop(c, None)
            # Not measured (tool not installed or stage skipped) is None, not zero.
            for c in ("messages", "received"):
                values[c] = values.get(c, 0) if (has_toxicity or has_sentiment) else None
            for c in ("toxic_emitted", "toxic_received"):
                values[c] = values.get(c, 0) if has_toxicity else None
            for c in ("negative_emitted", "negative_received"):
                values[c] = values.get(c, 0) if has_sentiment else None
            for c in ("threads", "toxic_threads"):
                values[c] = values.get(c, 0) if has_conv else None
            for c in ("psych_messages", "words", "we_words", "i_words", "thanks_emitted", "replies_received", "thanks_received"):
                values[c] = values.get(c, 0)  # dictionary counts: always measured
            for c in ("reacted_items", "reactions_pos_received", "reactions_neg_received"):
                values[c] = values.get(c, 0) if has_reactions else None  # mined before reactions were stored
            for c in EMOTION_COLUMNS:
                values[c] = values.get(c, 0) if has_emotions else None  # EmoTxt not installed or stage skipped
            return DeveloperMonth(project_id=pid, developer_id=dev_id, month=month, **values)

        ctx.db.execute(delete(DeveloperMonth).where(DeveloperMonth.project_id == pid))
        ctx.db.add_all(month_row(dev_id, month, values) for (dev_id, month), values in rows.items())
        assert set(strain_cols) <= {c.name for c in DeveloperMonth.__table__.columns}
        for dev in ctx.db.scalars(select(Developer).where(Developer.project_id == pid)):
            dev.first_activity = first.get(dev.id)
            dev.last_activity = last.get(dev.id)
        ctx.progress(len(rows), len(rows), f"{len(rows)} developer-months")
        ctx.log(f"Aggregated {len(rows)} developer-months")
