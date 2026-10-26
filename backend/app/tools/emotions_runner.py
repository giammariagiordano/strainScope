"""Emotions runner (RoBERTa). Executed with external/toxbert-venv, never imported by the backend.

Protocol: one JSON list per line on stdin, [{"id": ..., "text": ...}]; one JSON list per line on stdout,
[{"id": ..., "emotions": ["anger", ...]}] (empty list: no emotion). "READY" is printed at start-up.

Model: RoBERTa-base fine-tuned on Shaver's six emotions (Stack Overflow and GitHub gold standards), one sigmoid per
emotion; either a single model (train_roberta.py) or the 5-fold ensemble (train_roberta_cv.py, probabilities
averaged), with its decision thresholds (docs/EMOTIONS.md). Texts arrive cleaned by the backend (app.stages.text.clean_text), as in training.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MAX_LEN = 128  # as in training
model_dir = Path(sys.argv[1])
dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# A directory with fold_*/ sub-models is an ensemble (train_roberta_cv.py): probabilities are averaged.
paths = sorted(model_dir.glob("fold_*")) or [model_dir]
tok = AutoTokenizer.from_pretrained(paths[0])
models = [AutoModelForSequenceClassification.from_pretrained(p).to(dev).eval() for p in paths]
thresholds = json.loads((model_dir / "thresholds.json").read_text())
EMOTIONS = list(thresholds)  # anger, joy, love, sadness, fear, surprise: the order of the model's outputs
TH = np.array([thresholds[e] for e in EMOTIONS])
torch.set_grad_enabled(False)
print("READY", flush=True)


def classify(texts: list[str], bs: int = 32) -> np.ndarray:
    order = np.argsort([len(t) for t in texts])  # length-sorted batches waste less padding
    probs = np.empty((len(texts), len(EMOTIONS)))
    for b in range(0, len(order), bs):
        idx = order[b : b + bs]
        enc = tok([texts[i] for i in idx], truncation=True, max_length=MAX_LEN, padding=True, pad_to_multiple_of=16,
                  return_tensors="pt").to(dev)  # fmt: skip
        probs[idx] = np.mean([torch.sigmoid(m(**enc).logits).float().cpu().numpy() for m in models], axis=0)
        if dev.type == "mps":
            torch.mps.empty_cache()  # one graph per shape otherwise accumulates (see train_roberta.py)
    return probs


for line in sys.stdin:
    batch = json.loads(line)
    probs = classify([b["text"] for b in batch])
    print(json.dumps([{"id": b["id"], "emotions": [e for e, p, t in zip(EMOTIONS, row, TH) if p >= t]}
                      for b, row in zip(batch, probs)]), flush=True)  # fmt: skip
