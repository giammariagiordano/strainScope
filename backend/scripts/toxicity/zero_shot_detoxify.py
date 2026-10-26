"""Baseline 3: Detoxify's unbiased-toxic-roberta (Jigsaw-trained, Apache-2.0) used as is, no GitHub training.
Run from external/toxicity-gh with toxbert-venv. Scores = max over windows of the "toxicity" logit.
"""

import sys
import time
from pathlib import Path

import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import report  # noqa: E402
from windows import device, score_conversations  # noqa: E402

MODEL = "unitary/unbiased-toxic-roberta"
W = Path("work")
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL).to(device())
idx = model.config.label2id.get("toxicity", 0)
print("labels:", list(model.config.label2id)[:6], "... toxicity index", idx, "| device", device())

va, te = pd.read_parquet(W / "large_val.parquet"), pd.read_parquet(W / "large_test.parquet")
im = pd.read_parquet(W / "imran_conversations.parquet")
t0 = time.time()
prog = lambda done, total: print(f"   {done}/{total} windows, {time.time() - t0:.0f}s", flush=True)  # noqa: E731
s_va = score_conversations(model, tok, va.text.tolist(), idx, progress=prog)
s_te = score_conversations(model, tok, te.text.tolist(), idx, progress=prog)
s_im = score_conversations(model, tok, im.text.tolist(), idx)
pd.DataFrame({"issue_url": te.issue_url, "score": s_te}).to_parquet(W / "results/detoxify_test_scores.parquet")
report("Detoxify RoBERTa, zero-shot", (va.label, s_va), (te.label, s_te, te.repo), (im.label, s_im),
       out_path=W / "results/detoxify_zero_shot.json")  # fmt: skip
print(f"done in {time.time() - t0:.0f}s")
