"""Validate the EmoTxt models on GitHub: the gold standard of Novielli et al. (MSR 2020, 7,122 pull-request and
commit comments) was annotated with Shaver's emotions and then mapped to polarity (love, joy → positive; anger,
sadness, fear → negative; surprise dropped). Applying the same mapping to our predictions gives a GitHub validation
of the emotion classifiers without new annotations.

Run from external/ with external/emotions-venv:
    emotions-venv/bin/python ../backend/scripts/emotions/validate_github.py [emotxt|roberta]

Prediction → polarity: positive if only love/joy, negative if only anger/sadness/fear, neutral if none,
"mixed" (counted as wrong) if both. Writes emotxt-model/github_validation.json.
"""

import csv
import json
import subprocess
import sys
from pathlib import Path

from sklearn.metrics import classification_report, cohen_kappa_score

ROOT = Path(__file__).resolve().parents[2]
RUNNERS = {
    "emotxt": ["emotions-venv/bin/python", str(ROOT / "app" / "tools" / "emotxt_runner.py"), "emotxt", "emotxt-model"],
    "roberta": ["toxbert-venv/bin/python", str(ROOT / "app" / "tools" / "emotions_runner.py"), "roberta-emotions"],
    "roberta-cv": ["toxbert-venv/bin/python", str(ROOT / "app" / "tools" / "emotions_runner.py"), "roberta-emotions-cv"],
    "roberta-final": ["toxbert-venv/bin/python", str(ROOT / "app" / "tools" / "emotions_runner.py"), "roberta-emotions-final"],
}
POSITIVE, NEGATIVE = {"love", "joy"}, {"anger", "sadness", "fear"}


def polarity(emotions: list[str]) -> str:
    pos, neg = bool(POSITIVE & set(emotions)), bool(NEGATIVE & set(emotions))
    return "mixed" if pos and neg else "positive" if pos else "negative" if neg else "neutral"


def main():
    with open("gold/github_gold.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter=";"))
    which = sys.argv[1] if len(sys.argv) > 1 else "roberta"
    proc = subprocess.Popen(RUNNERS[which], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    assert proc.stdout.readline().strip() == "READY"
    predicted = []
    for i in range(0, len(rows), 2000):
        batch = [{"id": r["ID"], "text": r["Text"]} for r in rows[i : i + 2000]]
        proc.stdin.write(json.dumps(batch) + "\n")
        proc.stdin.flush()
        predicted += json.loads(proc.stdout.readline())
    proc.stdin.close()
    proc.wait()

    gold = [r["Polarity"] for r in rows]
    pred = [polarity(p["emotions"]) for p in predicted]
    report = classification_report(gold, pred, labels=["positive", "neutral", "negative"], output_dict=True, zero_division=0)
    out = {
        "n": len(rows),
        "macro_f1": round(report["macro avg"]["f1-score"], 3),
        "per_class": {k: {m: round(report[k][m], 3) for m in ("precision", "recall", "f1-score")} for k in ("positive", "neutral", "negative")},
        "kappa": round(cohen_kappa_score(gold, pred), 3),
        "mixed_predictions": pred.count("mixed"),
        "emotion_rates": {e: round(sum(e in p["emotions"] for p in predicted) / len(rows), 4)
                          for e in ("anger", "joy", "love", "sadness", "fear", "surprise")},  # fmt: skip
    }
    print(json.dumps(out, indent=1))
    Path(f"{ {'emotxt': 'emotxt-model', 'roberta': 'roberta-emotions', 'roberta-cv': 'roberta-emotions-cv', 'roberta-final': 'roberta-emotions-final'}[which] }/github_validation.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
