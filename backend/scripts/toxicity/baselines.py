"""Baselines for conversation-level toxicity (docs/TOXICITY.md). Run from external/toxicity-gh with toxbert-venv.

1. LLM alone: Ministral-3-14B labels from the package, i.e. the labels BEFORE verifier and human review.
   Caveat: the reference labels were derived from these, so this baseline is optimistic.
2. TF-IDF (word 1-2 grams + char 3-5 grams) + logistic regression with balanced class weights.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import report  # noqa: E402

W = Path("work")
tr, va, te = (pd.read_parquet(W / f"large_{s}.parquet") for s in ("train", "val", "test"))
im = pd.read_parquet(W / "imran_conversations.parquet")
(W / "results").mkdir(exist_ok=True)

report("LLM alone (Ministral-3-14B, pre-review)",
       (va.label, va.llm_label.astype(float)), (te.label, te.llm_label.astype(float), te.repo),
       out_path=W / "results/llm_alone.json")  # fmt: skip

clip = lambda s: s.str.slice(0, 20000)  # noqa: E731  very long threads (max 390k chars) are truncated
word = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=300_000, sublinear_tf=True)
char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=5, max_features=300_000, sublinear_tf=True)
Xtr = hstack([word.fit_transform(clip(tr.text)), char.fit_transform(clip(tr.text))]).tocsr()
X = lambda d: hstack([word.transform(clip(d.text)), char.transform(clip(d.text))]).tocsr()  # noqa: E731
lr = LogisticRegression(C=4.0, class_weight="balanced", max_iter=3000, solver="liblinear").fit(Xtr, tr.label)
score = lambda d: lr.predict_proba(X(d))[:, 1]  # noqa: E731
report("TF-IDF + logistic regression", (va.label, score(va)), (te.label, score(te), te.repo), (im.label, score(im)),
       out_path=W / "results/tfidf_lr.json")  # fmt: skip
