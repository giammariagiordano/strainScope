"""Psycholinguistic markers from developers' messages (dictionary counts, no model).

- Belonging: first-person plural ("we", "our", "let's") versus singular ("I", "my") pronouns. In the
  psycholinguistic literature (Pennebaker; LIWC's *we* and *i* categories) a shift from "we" to "I" marks a
  weaker identification with the group.
- Gratitude: a message that thanks or appreciates someone ("thanks", "thank you", "appreciate", "great work",
  "nice catch", 🙏 …). Received gratitude is recognition, a resource in the JD-R model.

Messages are cleaned first (quoted replies, code, HTML comments and @mentions removed: app.stages.text.clean_text),
so a quoted "thanks" or a code variable named `we` does not count. English only.
"""

import re

WORD = re.compile(r"[a-z]+(?:'[a-z]+)?")
WE = {"we", "us", "our", "ours", "ourselves", "we're", "we've", "we'd", "we'll", "let's"}
I = {"i", "me", "my", "mine", "myself", "i'm", "i've", "i'd", "i'll"}  # noqa: E741
GRATITUDE = re.compile(
    r"\b(thanks?|thank\s+you|thankful|thx|tysm|appreciated?|appreciate\s+it|kudos|cheers"
    r"|(great|nice|good|awesome|amazing|excellent)\s+(work|job|catch|find|idea|stuff)"
    r"|well\s+done|much\s+appreciated|grateful)\b|🙏|❤️|💯",
    re.I,
)


def pronouns(text: str) -> tuple[int, int, int]:
    """(words, we-words, I-words) of a cleaned message."""
    words = WORD.findall(text.lower().replace("’", "'"))
    return len(words), sum(w in WE for w in words), sum(w in I for w in words)


def grateful(text: str) -> bool:
    return bool(GRATITUDE.search(text))


POSITIVE_REACTIONS = ("+1", "heart", "hooray", "rocket")
NEGATIVE_REACTIONS = ("-1", "confused")


def reactions(payload: dict | None) -> tuple[int | None, int | None]:
    """(positive, negative) from GitHub's REST `reactions` summary; None if the payload has none.
    "laugh" and "eyes" are left out: their tone depends on context."""
    if not payload:
        return None, None
    return sum(payload.get(k, 0) for k in POSITIVE_REACTIONS), sum(payload.get(k, 0) for k in NEGATIVE_REACTIONS)
