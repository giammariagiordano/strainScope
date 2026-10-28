"""Score an OpenAI model (app/tools/openai_emotions_runner.py) on the GitHub emotion gold standard of Imran et al.

Run from external/ with external/emotions-venv (texts are sent to OpenAI). The key is OPENAI_API_KEY or, if unset,
the one saved in the app's Settings:
    emotions-venv/bin/python ../backend/scripts/emotions/eval_openai.py [--model gpt-6-luna] [--shots 0] [--test]

By default it scores the 320-comment validation split of train_roberta.py (20% of GitHub-train, same seed), so that
the result is comparable with the local models' validation micro-F1 (RoBERTa 0.567) and the test set stays untouched.
--test scores the official 400-comment test set: run it once, for the configuration chosen on validation.
--shots k adds k labelled examples to the prompt, drawn from the other 80% of GitHub-train (never from the
validation or test comments); they are written to openai-emotions/examples.json, which the emotions stage then
uses as well, so the stage runs exactly the evaluated configuration. --shots 0 removes that file.
Answers are cached in openai-emotions/cache-<model>.jsonl (a re-run costs nothing). Results are appended to
openai-emotions/results.jsonl.
"""

import argparse
import json
import os
import subprocess
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
EMOTIONS = ["Anger", "Joy", "Love", "Sadness", "Fear", "Surprise"]  # gold columns
SEED = 13  # as in train_roberta.py
OUT = Path("openai-emotions")


def api_key() -> str:
    """OPENAI_API_KEY, else the key saved in Settings (read through the backend, which owns the database)."""
    if os.environ.get("OPENAI_API_KEY"):
        return os.environ["OPENAI_API_KEY"]
    code = "from app import emotion_settings; print(emotion_settings.current().api_key or '')"
    key = subprocess.run([str(ROOT / ".venv" / "bin" / "python"), "-c", code], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    if not key:
        raise SystemExit("No OpenAI key: save one in Settings, Emotion classifier, or set OPENAI_API_KEY")
    return key


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-6-luna")
    ap.add_argument("--shots", type=int, default=0)
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()

    gh = pd.read_csv("gh-emotions/gh_clean.csv").fillna({"text": ""})
    gh_train = gh[gh["split"] == "train"]
    fit, val = train_test_split(gh_train, test_size=0.2, random_state=SEED, stratify=gh_train["Anger"])
    scored = gh[gh["split"] == "test"] if args.test else val

    OUT.mkdir(exist_ok=True)
    examples = OUT / "examples.json"
    if args.shots:
        # half with at least one emotion, half without, so the model also sees neutral messages
        emotional = fit[fit[EMOTIONS].sum(axis=1) > 0].sample(args.shots - args.shots // 2, random_state=SEED)
        neutral = fit[fit[EMOTIONS].sum(axis=1) == 0].sample(args.shots // 2, random_state=SEED)
        shots = pd.concat([emotional, neutral]).sample(frac=1, random_state=SEED)
        examples.write_text(json.dumps([{"text": r["text"], "emotions": [e.lower() for e in EMOTIONS if r[e]]}
                                        for _, r in shots.iterrows()], indent=1))  # fmt: skip
    else:
        examples.unlink(missing_ok=True)

    cmd = [str(ROOT / ".venv" / "bin" / "python"), str(ROOT / "app" / "tools" / "openai_emotions_runner.py"),
           args.model, str(OUT / f"cache-{args.model}.jsonl")] + ([str(examples)] if args.shots else [])  # fmt: skip
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, env={**os.environ, "OPENAI_API_KEY": api_key()})
    assert proc.stdout.readline().strip() == "READY"
    proc.stdin.write(json.dumps([{"id": i, "text": t} for i, t in enumerate(scored["text"])]) + "\n")
    proc.stdin.flush()
    predicted = json.loads(proc.stdout.readline())
    proc.stdin.close()
    proc.wait()

    failed = sum(p["emotions"] is None for p in predicted)
    y = scored[EMOTIONS].to_numpy()
    yp = pd.DataFrame([[int(e.lower() in (p["emotions"] or [])) for e in EMOTIONS] for p in predicted]).to_numpy()
    prec, rec, f1, _ = precision_recall_fscore_support(y, yp, zero_division=0)
    res = {
        "model": args.model, "shots": args.shots, "split": "test" if args.test else "validation", "n": len(scored),
        "failed": failed, "micro_f1": round(f1_score(y.ravel(), yp.ravel()), 3),
        "per_emotion": {e.lower(): {"precision": round(p, 3), "recall": round(r, 3), "f1": round(f, 3)}
                        for e, p, r, f in zip(EMOTIONS, prec, rec, f1)},
    }  # fmt: skip
    print(json.dumps(res, indent=1))
    with open(OUT / "results.jsonl", "a") as f:
        f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
