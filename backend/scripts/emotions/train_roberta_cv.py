"""RoBERTa emotions, improved: 5-fold cross-fitting on GitHub-train with the polarity-augmented data of Imran et al.
(ASE 2022), decision thresholds from out-of-fold predictions on all 1,600 GitHub-train comments, and the 5 fold
models averaged on the test set. Compared with train_roberta.py (one model, thresholds from 320 comments) on the
same official GitHub test set.

Run from external/ with external/toxbert-venv:
    HF_HOME=$PWD/hf-cache PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.8 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.6 \
        toxbert-venv/bin/python ../backend/scripts/emotions/train_roberta_cv.py

Per fold k: train on SO (4,800) + the GitHub-train comments outside fold k + AUG_PER_COMMENT augmented variants of
each of them (variants of fold-k comments are never used for training: no leakage into the out-of-fold
predictions). Same model and optimisation as train_roberta.py; EPOCHS fixed at 4, the epoch train_roberta.py chose on
its validation split. The augmented data derive from GitHub-train comments only (checked: 0 of their ids are in test).

Outputs roberta-emotions-cv/: fold_{0..4}/ models, thresholds.json, results.json (out-of-fold and test scores, next
to the single model's and EmoTxt's).
"""

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_roberta import BASE, EMOTIONS, LR, MAX_LEN, BATCH, SEED, best_thresholds, device, predict, scores  # noqa: E402

EPOCHS, FOLDS, AUG_PER_COMMENT = 4, 5, 4
OUT = Path("roberta-emotions-cv")


def train_model(texts, labels, dev, tok):
    model = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=len(EMOTIONS), problem_type="multi_label_classification").to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    steps = EPOCHS * (len(texts) // BATCH + 1)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    lengths = np.array([min(len(tok(x)["input_ids"]), MAX_LEN) for x in texts], dtype=float)
    for epoch in range(EPOCHS):
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
        print(f"    epoch {epoch + 1}: {(time.time() - t0) / 60:.1f} min", flush=True)
    return model


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    dev = device()
    tok = AutoTokenizer.from_pretrained(BASE)
    so = pd.read_csv("gh-emotions/so_clean.csv").fillna({"text": ""})
    gh = pd.read_csv("gh-emotions/gh_clean.csv").fillna({"text": ""})
    aug = pd.read_csv("gh-emotions/aug_clean.csv").fillna({"text": ""})
    train, test = gh[gh["split"] == "train"].reset_index(drop=True), gh[gh["split"] == "test"].reset_index(drop=True)
    assert not set(aug["id"]) & set(test["id"]), "augmented data must not derive from test comments"
    # AUG_PER_COMMENT variants per original comment, excluding the copy identical to the original
    aug = aug.merge(train[["id", "text"]].rename(columns={"text": "original"}), on="id")
    aug = aug[aug["text"] != aug["original"]].groupby("id", group_keys=False).apply(
        lambda g: g.sample(min(len(g), AUG_PER_COMMENT), random_state=SEED))
    print(f"SO {len(so)}, GH-train {len(train)}, augmented variants {len(aug)}, test {len(test)}", flush=True)

    OUT.mkdir(exist_ok=True)
    oof = np.zeros((len(train), len(EMOTIONS)))
    test_probs = []
    folds = StratifiedKFold(FOLDS, shuffle=True, random_state=SEED).split(train, train["Anger"])
    for k, (fit, held) in enumerate(folds):
        t0 = time.time()
        ids = set(train.loc[fit, "id"])
        parts = [so, train.loc[fit], aug[aug["id"].isin(ids)]]
        texts = pd.concat([p["text"] for p in parts]).tolist()
        labels = pd.concat([p[EMOTIONS] for p in parts]).to_numpy(np.float32)
        print(f"fold {k}: {len(texts)} training texts", flush=True)
        model = train_model(texts, labels, dev, tok)
        oof[held] = predict(model, tok, train.loc[held, "text"].tolist(), dev)
        test_probs.append(predict(model, tok, test["text"].tolist(), dev))
        model.save_pretrained(OUT / f"fold_{k}")
        tok.save_pretrained(OUT / f"fold_{k}")
        del model
        if dev.type == "mps":
            torch.mps.empty_cache()
        print(f"fold {k} done in {(time.time() - t0) / 60:.0f} min", flush=True)

    y_train, y_test = train[EMOTIONS].to_numpy(), test[EMOTIONS].to_numpy()
    th = np.array(best_thresholds(y_train, oof))
    oof_scores = scores(y_train, (oof >= th).astype(int))
    pt = np.mean(test_probs, axis=0)
    test_scores = scores(y_test, (pt >= th).astype(int))
    single = json.loads(Path("roberta-emotions/results.json").read_text())
    results = {"cv_ensemble": {"thresholds": dict(zip([e.lower() for e in EMOTIONS], th.tolist())),
                               "out_of_fold": oof_scores, "gh_test": test_scores,
                               "setup": {"folds": FOLDS, "epochs": EPOCHS, "aug_per_comment": AUG_PER_COMMENT}},
               "single_model": single["roberta"]["gh_test"], "emotxt_so+gh": single["emotxt_so+gh"]}  # fmt: skip
    (OUT / "thresholds.json").write_text(json.dumps(results["cv_ensemble"]["thresholds"]))
    (OUT / "results.json").write_text(json.dumps(results, indent=1))
    np.save(OUT / "test_probs.npy", pt)
    for name, r in (("CV ensemble", test_scores), ("single model", results["single_model"]), ("EmoTxt", results["emotxt_so+gh"])):
        print(f"{name:13s} test micro-F1 {r['micro_f1']}", {e: v["f1"] for e, v in r["per_emotion"].items()})


if __name__ == "__main__":
    main()
