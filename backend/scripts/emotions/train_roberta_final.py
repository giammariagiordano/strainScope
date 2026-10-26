"""Deployed emotion model: one RoBERTa trained like each fold of train_roberta_cv.py, but on all of GitHub-train
(SO + 1,600 comments + their augmented variants), with the thresholds chosen on the out-of-fold predictions of the
5-fold run. The GitHub test set is still unseen, so its score is an honest estimate. One model is five times faster
than the 5-fold ensemble, which it matches closely (single fold models: micro-F1 0.612-0.645; ensemble 0.632).

Run from external/ with external/toxbert-venv after train_roberta_cv.py:
    HF_HOME=$PWD/hf-cache PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.8 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.6 \
        toxbert-venv/bin/python ../backend/scripts/emotions/train_roberta_final.py
"""

import json
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_roberta import BASE, EMOTIONS, SEED, device, predict, scores  # noqa: E402
from train_roberta_cv import AUG_PER_COMMENT, train_model  # noqa: E402

OUT = Path("roberta-emotions-final")


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
    aug = aug.merge(train[["id", "text"]].rename(columns={"text": "original"}), on="id")
    aug = aug[aug["text"] != aug["original"]].groupby("id", group_keys=False).apply(
        lambda g: g.sample(min(len(g), AUG_PER_COMMENT), random_state=SEED))
    parts = [so, train, aug]
    texts = pd.concat([p["text"] for p in parts]).tolist()
    labels = pd.concat([p[EMOTIONS] for p in parts]).to_numpy(np.float32)
    print(f"training on {len(texts)} texts", flush=True)
    model = train_model(texts, labels, dev, tok)
    model.save_pretrained(OUT)
    tok.save_pretrained(OUT)
    shutil.copy("roberta-emotions-cv/thresholds.json", OUT / "thresholds.json")
    th = np.array(list(json.loads((OUT / "thresholds.json").read_text()).values()))
    test_scores = scores(test[EMOTIONS].to_numpy(), (predict(model, tok, test["text"].tolist(), dev) >= th).astype(int))
    cv = json.loads(Path("roberta-emotions-cv/results.json").read_text())
    (OUT / "results.json").write_text(json.dumps({"final_single_model": test_scores, **cv}, indent=1))
    print("final single model test micro-F1", test_scores["micro_f1"], {e: v["f1"] for e, v in test_scores["per_emotion"].items()})


if __name__ == "__main__":
    main()
