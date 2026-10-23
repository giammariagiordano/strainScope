"""Senti4SD runner. Executed with Senti4SD's own interpreter (external/senti4sd-venv), never imported by the backend.

Protocol: one JSON list per line on stdin, [{"id": ..., "text": "..."}]; one JSON list per line on stdout,
[{"id": ..., "polarity": "positive" | "neutral" | "negative"}]. "READY" is printed at start-up.

Pipeline, as in pySenti4SD's classification.sh: Senti4SD-fast.jar extracts the features (lexicon, keyword
and semantic features over the Stack Overflow word embeddings), then the linear model predicts the polarity.
The model is Senti4SD's learner (L1-regularized logistic regression, C=0.5) retrained on the GitHub gold
standard of Novielli et al. (MSR 2020): 10-fold macro-F1 0.89 on GitHub, against 0.70 for the published
Stack Overflow model (see README). Prediction is done in numpy (senti4sd_model.py), equivalent to LIBLINEAR.
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from senti4sd_model import LiblinearModel  # noqa: E402

repo = Path(sys.argv[1])
model = LiblinearModel(repo / os.environ.get("SENTI4SD_MODEL", "Senti4SD-github.model"))
JAVA = [
    "java", "-Duser.language=en", "-Duser.country=US", "-Dfile.encoding=UTF-8",  # locale changes tokenization
    "-jar", str(repo / "java" / "Senti4SD-fast.jar"), "-F", "A", "-W", str(repo / "java" / "dsm.bin"), "-vd", "600",
]  # fmt: skip
print("READY", flush=True)

for line in sys.stdin:
    batch = json.loads(line)
    with tempfile.TemporaryDirectory() as tmp:
        src, out = Path(tmp) / "input.txt", Path(tmp) / "features.csv"
        # One document per line, the format pySenti4SD's csv_processing.py produces for the jar.
        src.write_text("\n".join(" ".join(b["text"].split()) for b in batch) + "\n", encoding="utf-8")
        proc = subprocess.run([*JAVA, "-i", str(src), "-oc", str(out)], capture_output=True, text=True)
        if proc.returncode != 0 or not out.exists():
            sys.stderr.write(proc.stderr[-2000:])
            sys.exit(1)
        features = pd.read_csv(out)
    if len(features) != len(batch):
        sys.stderr.write(f"feature rows {len(features)} != batch {len(batch)}\n")
        sys.exit(1)
    polarity = model.predict(features.drop(columns=["id"]).to_numpy(np.float32))
    print(json.dumps([{"id": b["id"], "polarity": str(p)} for b, p in zip(batch, polarity)]), flush=True)
