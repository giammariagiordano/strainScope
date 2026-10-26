"""Prepare the toxicity training/evaluation data from the ASE 2026 replication package (Zenodo 10.5281/zenodo.21231251).

Run from external/toxicity-gh with the backend interpreter:
    ../../backend/.venv/bin/python ../../backend/scripts/toxicity/prepare_data.py

Outputs (external/toxicity-gh/work/):
    large_{train,val,test}.parquet   124,757 conversations split BY REPOSITORY (70/15/15, seed 13), so the test set
                                     contains only repositories never seen in training, at the natural toxicity rate.
    imran_conversations.parquet      898 conversations labelled entirely by humans (external test set).
    stats.json                       dataset statistics reported in docs/TOXICITY.md.

Conversations are the package's text ("user: comment" lines). Known user names (from the package's comment tables)
are replaced by speaker placeholders ("P1:", "P2:", ...) so that the model cannot learn from who is speaking, only
from what is said.
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

BASE = Path("backup/dataset")
OUT = Path("work")
OUT.mkdir(exist_ok=True)
SEED = 13


def anonymizer(users: list[str]):
    """Replace the conversation's known user names at line starts with P1, P2, ... in order of appearance.

    Only names that appear in the package's comment table for that conversation are replaced, so lines such as
    "Note:" or "[https://..." are left untouched.
    """
    names = sorted({u for u in users if isinstance(u, str) and u}, key=len, reverse=True)
    if not names:
        return lambda text: text or ""
    pattern = re.compile(r"^(" + "|".join(re.escape(n) for n in names) + r"):", re.M)

    def apply(text: str) -> str:
        ids: dict[str, str] = {}

        def sub(m):
            ids.setdefault(m.group(1), f"P{len(ids) + 1}")
            return f"{ids[m.group(1)]}:"

        return pattern.sub(sub, text or "")

    return apply


MENTION = re.compile(r"(?<![\w.])@[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})")


def normalize_mentions(text: str) -> str:
    """@mentions are user names: map them to one token, as StrainScope does before classifying."""
    return MENTION.sub("@user", text)


def repo_of(url: str) -> str:
    """owner/name from an API URL (api.github.com/repos/o/r/issues/n) or a web URL (github.com/o/r/pull/n)."""
    path = url.split("/repos/", 1)[1] if "/repos/" in url else url.split("github.com/", 1)[1]
    owner, name = path.split("/")[:2]
    return f"{owner}/{name}".lower()


labels = pd.read_csv(BASE / "large-dataset/toxicity_label_complete_124k.csv")
conv = pd.read_csv(BASE / "large-dataset/overall_issue_conversations_124k.csv")
df = conv.merge(labels, on="issue_url", how="inner")
df["repo"] = df["issue_url"].map(repo_of)
users = pd.read_csv(BASE / "large-dataset/overall_comments_124k.csv", usecols=["issue_url", "user"])
users_by_issue = users.groupby("issue_url")["user"].apply(list).to_dict()
df["text"] = [normalize_mentions(anonymizer(users_by_issue.get(u, []))(t)) for u, t in zip(df["issue_url"], df["conversation"].fillna(""))]
df["label"] = df["annotated_toxicity"].astype(int)
df["llm_label"] = df["toxicity_ministral"].astype(int)
df = df[["issue_url", "repo", "text", "label", "llm_label", "is_pull_req", "comments_count", "created_at"]]

# Repository-level split: test repositories are unseen during training and threshold selection.
gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=SEED)
train_idx, rest_idx = next(gss.split(df, groups=df["repo"]))
rest = df.iloc[rest_idx]
gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=SEED)
val_idx, test_idx = next(gss2.split(rest, groups=rest["repo"]))
splits = {"train": df.iloc[train_idx], "val": rest.iloc[val_idx], "test": rest.iloc[test_idx]}
for name, part in splits.items():
    part.reset_index(drop=True).to_parquet(OUT / f"large_{name}.parquet")

imran = pd.read_csv(BASE / "derailment-paper-data/unified_final_dataset_conversations.csv")
imran_comments = pd.read_csv(BASE / "derailment-paper-data/unified_final_dataset_comments.csv", usecols=["issue_id", "speaker"])
speakers = imran_comments.groupby("issue_id")["speaker"].apply(list).to_dict()
imran = pd.DataFrame({"issue_id": imran["issue_id"],
                      "text": [normalize_mentions(anonymizer(speakers.get(i, []))(t)) for i, t in zip(imran["issue_id"], imran["speaker_text"].fillna(""))],
                      "label": imran["is_toxic"].astype(int)})  # fmt: skip
imran.to_parquet(OUT / "imran_conversations.parquet")

# Overlap with the repositories mined by StrainScope (to rule out contamination of later analyses).
ours = set()
db = Path("../../backend/data/strainscope.db")
if db.exists():
    import sqlite3

    ours = {r[0].lower() for r in sqlite3.connect(db).execute("select owner || '/' || name from projects")}

chars = df["text"].str.len()
stats = {
    "conversations": len(df),
    "repositories": int(df["repo"].nunique()),
    "toxic": int(df["label"].sum()),
    "toxic_rate": round(float(df["label"].mean()), 5),
    "llm_toxic": int(df["llm_label"].sum()),
    "llm_vs_annotated": {
        "agree": int((df["llm_label"] == df["label"]).sum()),
        "llm_1_annotated_0": int(((df["llm_label"] == 1) & (df["label"] == 0)).sum()),
        "llm_0_annotated_1": int(((df["llm_label"] == 0) & (df["label"] == 1)).sum()),
    },
    "pull_requests": int(df["is_pull_req"].sum()),
    "chars": {"median": int(chars.median()), "p90": int(chars.quantile(0.9)), "max": int(chars.max())},
    "splits": {name: {"conversations": len(p), "repositories": int(p["repo"].nunique()), "toxic": int(p["label"].sum()),
                      "toxic_rate": round(float(p["label"].mean()), 5)} for name, p in splits.items()},  # fmt: skip
    "imran": {"conversations": len(imran), "toxic": int(imran["label"].sum())},
    "overlap_with_strainscope_projects": sorted(ours & set(df["repo"])),
    "seed": SEED,
}
(OUT / "stats.json").write_text(json.dumps(stats, indent=2))
print(json.dumps(stats, indent=2))
