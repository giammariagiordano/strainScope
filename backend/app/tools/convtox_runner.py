"""Conversation-toxicity runner. Executed with external/toxbert-venv, never imported by the backend.

Protocol: one JSON list per line on stdin, [{"id": ..., "text": "P1: ...\\nP2: ..."}]; one JSON list per line on
stdout, [{"id": ..., "p": logit, "toxic": 0|1}]. "READY" is printed at start-up.

Model: RoBERTa fine-tuned on the ASE 2026 GitHub conversations (docs/TOXICITY.md), max over 512-token windows,
decision threshold chosen on the validation split at the natural toxicity rate.
"""

import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_dir = Path(sys.argv[1])
sys.path.insert(0, sys.argv[2])  # backend/scripts/toxicity, for the shared windowing code
from windows import device, score_conversations  # noqa: E402

tok = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device())
threshold = json.loads((model_dir / "threshold.json").read_text())["threshold"]
torch.set_grad_enabled(False)
print("READY", flush=True)

for line in sys.stdin:
    batch = json.loads(line)
    scores = score_conversations(model, tok, [b["text"] for b in batch])
    print(json.dumps([{"id": b["id"], "p": round(float(s), 4), "toxic": int(s >= threshold)} for b, s in zip(batch, scores)]), flush=True)
