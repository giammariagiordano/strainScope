"""ToxiCR runner. Executed with ToxiCR's own interpreter (external/toxicr-venv), never imported by the backend.

Protocol: one JSON object per line on stdin, {"id": ..., "text": "..."}; one JSON object per line on stdout,
{"id": ..., "toxic": 0|1, "p": float}. The line "READY" is printed once the model is loaded.

Configuration: ToxiCR's random forest on TF-IDF with programming keywords removed
(`--algo RF --embed tfidf --keyword`), trained on ToxiCR's full code-review dataset with `--mode pretrain`.
"""

import json
import os
import sys

toxicr_dir = sys.argv[1]
sys.path.insert(0, toxicr_dir)
os.chdir(toxicr_dir)

import pandas as pd  # noqa: E402
from ToxiCR import ToxiCR  # noqa: E402

clf = ToxiCR(ALGO="RF", embedding="tfidf", remove_keywords=True, count_profanity=False, load_pretrained=True)
clf.init_predictor()
model = clf.classifier_model
toxic_col = list(model.model.classes_).index(1)
print("READY", flush=True)

for line in sys.stdin:
    batch = json.loads(line)  # list of {"id", "text"}
    frame = pd.DataFrame({"message": [b["text"] for b in batch]})
    clf.preprocess(frame)
    X = model.mapper.transform(frame[["message", "profane_count", "anger_count", "emoticon_count"]])
    probs = model.model.predict_proba(X)[:, toxic_col]
    out = [{"id": b["id"], "toxic": int(p >= 0.5), "p": round(float(p), 4)} for b, p in zip(batch, probs)]
    print(json.dumps(out), flush=True)
