"""Fine-tune RoBERTa-base for the six Shaver emotions and compare it with EmoTxt on the same GitHub test set.

Run from external/ with external/toxbert-venv (PyTorch, transformers):
    PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.5 \
        toxbert-venv/bin/python ../backend/scripts/emotions/train_roberta.py

Data (same as train_emotxt.py, cleaned like the pipeline): Stack Overflow gold standard (gh-emotions/so_clean.csv,
4,800) and GitHub comments (gh-emotions/gh_clean.csv, 2,000, official 1,600/400 split, Imran et al., ASE 2022).

Protocol: train on SO + 80% of GH-train; the other 20% of GH-train (stratified, seed 13) chooses the epoch and a
decision threshold per emotion (max F1); the official GH test set is used once. Multi-label head (one sigmoid per
emotion, binary cross-entropy), lr 2e-5, weight decay 0.01, 6% warm-up, linear decay,
max 128 tokens (95% of the texts are shorter), length-bucketed batches of 16 padded to multiples
of 16 (few distinct shapes: the MPS backend caches one graph per shape and runs out of memory otherwise), up to 6
epochs, seed 13.

Outputs roberta-emotions/: the model, thresholds.json, results.json (test micro-F1 and per emotion, next to EmoTxt's
from emotxt-model/results.json).
"""

import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

EMOTIONS = ["Anger", "Joy", "Love", "Sadness", "Fear", "Surprise"]
BASE = "FacebookAI/roberta-base"
OUT = Path("roberta-emotions")
SEED, MAX_LEN, BATCH, ACCUM, EPOCHS, LR = 13, 128, 16, 1, 6, 2e-5


def seed_all():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


@torch.no_grad()
def predict(model, tok, texts, dev, bs=16):
    model.eval()
    out = []
    order = np.argsort([len(t) for t in texts])  # length-sorted batches waste less padding
    probs = np.empty((len(texts), len(EMOTIONS)))
    for b in range(0, len(order), bs):
        idx = order[b : b + bs]
        enc = tok([texts[i] for i in idx], truncation=True, max_length=MAX_LEN, padding=True, pad_to_multiple_of=16, return_tensors="pt").to(dev)
        probs[idx] = torch.sigmoid(model(**enc).logits).float().cpu().numpy()
        if dev.type == "mps":
            torch.mps.empty_cache()
    return probs


def best_thresholds(y, p):
    grid = np.arange(0.05, 0.96, 0.05)
    return [float(max(grid, key=lambda t: f1_score(y[:, j], p[:, j] >= t, zero_division=0))) for j in range(len(EMOTIONS))]


def scores(y, pred):
    per = {}
    for j, e in enumerate(EMOTIONS):
        p, r, f, _ = precision_recall_fscore_support(y[:, j], pred[:, j], average="binary", zero_division=0)
        per[e.lower()] = {"precision": round(p, 3), "recall": round(r, 3), "f1": round(f, 3), "positives": int(y[:, j].sum())}
    return {"micro_f1": round(f1_score(y.ravel(), pred.ravel(), zero_division=0), 3), "per_emotion": per}


def main():
    seed_all()
    dev = device()
    so = pd.read_csv("gh-emotions/so_clean.csv").fillna({"text": ""})
    gh = pd.read_csv("gh-emotions/gh_clean.csv").fillna({"text": ""})
    gh_train, gh_test = gh[gh["split"] == "train"], gh[gh["split"] == "test"]
    fit_gh, val = train_test_split(gh_train, test_size=0.2, random_state=SEED, stratify=gh_train["Anger"])
    train = pd.concat([so, fit_gh], ignore_index=True)
    print(f"train {len(train)} (SO {len(so)} + GH {len(fit_gh)}), validation {len(val)}, test {len(gh_test)}", flush=True)

    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=len(EMOTIONS), problem_type="multi_label_classification").to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    steps = EPOCHS * (len(train) // (BATCH * ACCUM) + 1)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    X, Y = train["text"].tolist(), train[EMOTIONS].to_numpy(np.float32)
    lengths = np.array([min(len(tok(x)["input_ids"]), MAX_LEN) for x in X], dtype=float)
    yv = val[EMOTIONS].to_numpy()

    best, history = None, []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        t0 = time.time()
        # Length-bucketed batches (shuffled within and across buckets): far less padding than random batches.
        by_len = np.argsort(lengths + np.random.rand(len(X)))
        batches = [by_len[b : b + BATCH] for b in range(0, len(by_len), BATCH)]
        random.shuffle(batches)
        for k, idx in enumerate(batches):
            enc = tok([X[i] for i in idx], truncation=True, max_length=MAX_LEN, padding=True, pad_to_multiple_of=16, return_tensors="pt").to(dev)
            loss = model(**enc, labels=torch.tensor(Y[idx], device=dev)).loss / ACCUM
            loss.backward()
            if (k + 1) % ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
            if dev.type == "mps":
                torch.mps.empty_cache()
        pv = predict(model, tok, val["text"].tolist(), dev)
        th = best_thresholds(yv, pv)
        f = f1_score(yv.ravel(), (pv >= np.array(th)).ravel(), zero_division=0)
        history.append({"epoch": epoch, "val_micro_f1": round(f, 3), "minutes": round((time.time() - t0) / 60, 1)})
        print(history[-1], flush=True)
        if best is None or f > best[0]:
            best = (f, epoch, th)
            model.save_pretrained(OUT)
            tok.save_pretrained(OUT)

    f, epoch, th = best
    model = AutoModelForSequenceClassification.from_pretrained(OUT).to(dev)
    pt = predict(model, tok, gh_test["text"].tolist(), dev)
    test = scores(gh_test[EMOTIONS].to_numpy(), (pt >= np.array(th)).astype(int))
    emotxt = json.loads(Path("emotxt-model/results.json").read_text())
    results = {"roberta": {"best_epoch": epoch, "val_micro_f1": round(f, 3), "thresholds": dict(zip([e.lower() for e in EMOTIONS], th)),
                           "gh_test": test, "history": history},
               "emotxt_so+gh": emotxt["so+gh"]["gh_test"]}  # fmt: skip
    (OUT / "thresholds.json").write_text(json.dumps(results["roberta"]["thresholds"]))
    (OUT / "results.json").write_text(json.dumps(results, indent=1))
    print("RoBERTa test", test["micro_f1"], {e: v["f1"] for e, v in test["per_emotion"].items()})
    print("EmoTxt  test", emotxt["so+gh"]["gh_test"]["micro_f1"], {e: v["f1"] for e, v in emotxt["so+gh"]["gh_test"]["per_emotion"].items()})


if __name__ == "__main__":
    main()
