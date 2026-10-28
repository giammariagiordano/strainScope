"""Screen emotion-model variants on the validation split only (the GitHub test set is never used here).

Run from external/ with external/toxbert-venv:
    HF_HOME=$PWD/hf-cache PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.8 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.6 \
        toxbert-venv/bin/python ../backend/scripts/emotions/screen_models.py <name> <base model> [--goemotions]

Every variant trains on SO + 80% of GitHub-train + 4 augmented variants of each of those comments and is scored on
the remaining 20% of GitHub-train (the split of train_roberta.py), with per-emotion thresholds tuned on it; 4 epochs.
With --goemotions, the model is first fine-tuned for one epoch on GoEmotions (Demszky et al., ACL 2020; 43,410
Reddit comments), its 27 labels mapped to Shaver's six emotions (GOEMOTIONS_MAP), before the software data.
Results are appended to emotion-screening.jsonl; the best variant is then trained with train_roberta_cv.py's
protocol and scored once on the test set.
"""

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_roberta import BATCH, EMOTIONS, LR, MAX_LEN, SEED, best_thresholds, device, predict, scores  # noqa: E402

EPOCHS, AUG_PER_COMMENT = 4, 4
# Shaver et al.'s six basic emotions and their subordinate categories (irritation and exasperation under anger,
# affection under love, cheerfulness and pride under joy, disappointment and shame under sadness, nervousness under
# fear). Labels without a clear Shaver counterpart (approval, curiosity, confusion, desire, neutral, ...) are dropped.
GOEMOTIONS_MAP = {
    "Anger": ["anger", "annoyance", "disgust"],
    "Joy": ["joy", "amusement", "excitement", "gratitude", "optimism", "relief", "pride"],
    "Love": ["love", "caring", "admiration"],
    "Sadness": ["sadness", "disappointment", "grief", "remorse", "embarrassment"],
    "Fear": ["fear", "nervousness"],
    "Surprise": ["surprise", "realization"],
}


def goemotions() -> tuple[list[str], np.ndarray]:
    names = Path("goemotions/emotions.txt").read_text().split()
    d = pd.read_csv("goemotions/train.tsv", sep="\t", header=None, names=["text", "labels", "id"])
    y = np.zeros((len(d), len(EMOTIONS)), dtype=np.float32)
    for i, labels in enumerate(d["labels"].astype(str)):
        present = {names[int(k)] for k in labels.split(",")}
        for j, e in enumerate(EMOTIONS):
            y[i, j] = float(bool(present & set(GOEMOTIONS_MAP[e])))
    return d["text"].tolist(), y


def train(model, tok, texts, labels, dev, epochs):
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    steps = epochs * (len(texts) // BATCH + 1)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    lengths = np.array([min(len(tok(x)["input_ids"]), MAX_LEN) for x in texts], dtype=float)
    for epoch in range(epochs):
        model.train()
        t0 = time.time()
        by_len = np.argsort(lengths + np.random.rand(len(texts)))
        batches = [by_len[b : b + BATCH] for b in range(0, len(by_len), BATCH)]
        random.shuffle(batches)
        for idx in batches:
            enc = tok([texts[i] for i in idx], truncation=True, max_length=MAX_LEN, padding=True, pad_to_multiple_of=16,
                      return_tensors="pt").to(dev)  # fmt: skip
            loss = model(**enc, labels=torch.tensor(labels[idx], device=dev)).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            if dev.type == "mps":
                torch.mps.empty_cache()
        print(f"    epoch {epoch + 1}/{epochs}: {(time.time() - t0) / 60:.1f} min", flush=True)


def main():
    name, base = sys.argv[1], sys.argv[2]
    use_go = "--goemotions" in sys.argv
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    dev = device()
    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForSequenceClassification.from_pretrained(base, num_labels=len(EMOTIONS), problem_type="multi_label_classification").to(dev)

    so = pd.read_csv("gh-emotions/so_clean.csv").fillna({"text": ""})
    gh = pd.read_csv("gh-emotions/gh_clean.csv").fillna({"text": ""})
    aug = pd.read_csv("gh-emotions/aug_clean.csv").fillna({"text": ""})
    gh_train = gh[gh["split"] == "train"]
    fit, val = train_test_split(gh_train, test_size=0.2, random_state=SEED, stratify=gh_train["Anger"])
    aug = aug.merge(fit[["id", "text"]].rename(columns={"text": "original"}), on="id")
    aug = aug[aug["text"] != aug["original"]].groupby("id", group_keys=False).apply(
        lambda g: g.sample(min(len(g), AUG_PER_COMMENT), random_state=SEED))
    parts = [so, fit, aug]
    texts = pd.concat([p["text"] for p in parts]).tolist()
    labels = pd.concat([p[EMOTIONS] for p in parts]).to_numpy(np.float32)

    t0 = time.time()
    if use_go:
        gt, gy = goemotions()
        print(f"{name}: GoEmotions stage on {len(gt)} texts", flush=True)
        train(model, tok, gt, gy, dev, 1)
    print(f"{name}: software stage on {len(texts)} texts, validation {len(val)}", flush=True)
    train(model, tok, texts, labels, dev, EPOCHS)
    pv = predict(model, tok, val["text"].tolist(), dev)
    yv = val[EMOTIONS].to_numpy()
    th = np.array(best_thresholds(yv, pv))
    res = {"name": name, "base": base, "goemotions": use_go, "minutes": round((time.time() - t0) / 60, 1),
           "val_micro_f1_tuned": round(f1_score(yv.ravel(), (pv >= th).ravel()), 3),
           "val_micro_f1_at_0.5": round(f1_score(yv.ravel(), (pv >= 0.5).ravel()), 3),
           "val": scores(yv, (pv >= th).astype(int))}  # fmt: skip
    with open("emotion-screening.jsonl", "a") as f:
        f.write(json.dumps(res) + "\n")
    print(name, "validation micro-F1", res["val_micro_f1_tuned"], "(at 0.5:", res["val_micro_f1_at_0.5"], ")", flush=True)


if __name__ == "__main__":
    main()
