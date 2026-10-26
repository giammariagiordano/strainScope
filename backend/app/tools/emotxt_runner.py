"""EmoTxt runner (baseline, see docs/EMOTIONS.md; the emotions stage uses emotions_runner.py, RoBERTa). Executed with external/emotions-venv, never imported by the backend.

Protocol: one JSON list per line on stdin, [{"id": ..., "text": ...}]; one JSON list per line on stdout,
[{"id": ..., "emotions": ["anger", ...]}] (empty list: no emotion). "READY" is printed at start-up.

Texts are tokenized by EmoTxt's own Java pre-processing (one JVM per batch, `createDocFormat` mode), then the
features (emotxt_features.py, verified identical to the Java ones) and the six linear models trained by
backend/scripts/emotions/train_emotxt.py are applied in Python.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from emotxt_features import EmoTxtFeatures, java_preprocess  # noqa: E402

emotxt_dir, model_dir = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
model = np.load(model_dir / "model.npz")
EMOTIONS = [str(e) for e in model["emotions"]]
W = np.column_stack([model[f"w_{e}"] for e in EMOTIONS])
B = np.array([float(model[f"b_{e}"][0]) for e in EMOTIONS])
fx = EmoTxtFeatures(model_dir / "n-grams", model_dir / "WordnetCategories", json.loads((model_dir / "idf.json").read_text()))


print("READY", flush=True)
for line in sys.stdin:
    batch = json.loads(line)
    scores = fx.transform(java_preprocess([b["text"] for b in batch], emotxt_dir)) @ W + B
    print(json.dumps([{"id": b["id"], "emotions": [e for e, s in zip(EMOTIONS, row) if s > 0]}
                      for b, row in zip(batch, np.asarray(scores))]), flush=True)  # fmt: skip
