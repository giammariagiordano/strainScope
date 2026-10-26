"""Train EmoTxt's emotion classifiers (features reimplemented in app/tools/emotxt_features.py, verified identical to
the Java ones) and choose the training corpus for GitHub.

Run from external/ with external/emotions-venv:
    emotions-venv/bin/python ../backend/scripts/emotions/train_emotxt.py

Corpora
- SO: EmoTxt's Stack Overflow gold standard (Novielli et al., MSR 2018; 4,800 posts), pre-processed by the Java
  tokenizer in emotxt/training_so_anger_anger/ (one EmoTxt run).
- GH: GitHub comments annotated with the same six emotions (Imran et al., ASE 2022; 2,000 comments, official
  1,600/400 train/test split; github.com/vcu-swim-lab/SE-Emotion-Study), cleaned like the pipeline
  (gh-emotions/gh_clean.csv) and pre-processed by the same Java tokenizer.

Configurations (dictionary, IDF and models always fitted on the training data only):
  so      trained on SO (the original EmoTxt setting);
  gh      trained on GH-train;
  so+gh   trained on SO + GH-train.
Selection: 5-fold CV on GH-train (micro-F1 over the six emotions); the official GH test set is used once, to report.
Learner per emotion: LIBLINEAR L2 logistic regression or L2-loss SVM, cost by inner 3-fold CV on F1, balanced
class weights (EmoTxt tunes LIBLINEAR with caret and down-samples).

Outputs emotxt-model/: model.npz, idf.json, n-grams/, WordnetCategories/, results.json; the deployed model is the
selected configuration refitted on SO and/or all 2,000 GH comments.
"""

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "app" / "tools"))
from emotxt_features import EmoTxtFeatures, java_preprocess, write_dictionary  # noqa: E402

EMOTIONS = ["anger", "joy", "love", "sadness", "fear", "surprise"]
WORDNET = Path("emotxt/java/res/WordnetCategories")
OUT = Path("emotxt-model")
COSTS = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]


def so_corpus():
    run = Path("emotxt/training_so_anger_anger")
    texts = run.joinpath("ElaboratedFiles/onlyText_PreProcessed.txt").read_text(encoding="utf-8").split("\n")[:4800]
    y = {}
    for e in EMOTIONS:
        rows = Path(f"emotxt/so_{e}.csv").read_text(encoding="utf-8").splitlines()[1:]
        y[e] = np.array([r.split(";", 2)[1] == "YES" for r in rows], dtype=int)
    return texts, pd.DataFrame(y)


def gh_corpus():
    d = pd.read_csv("gh-emotions/gh_clean.csv").fillna({"text": ""})
    cache = Path("gh-emotions/gh_preprocessed.json")
    if cache.exists():
        texts = json.loads(cache.read_text())
    else:
        texts = java_preprocess(d["text"].tolist(), Path("emotxt"))
        cache.write_text(json.dumps(texts))
    y = pd.DataFrame({e: d[e.capitalize()].astype(int).to_numpy() for e in EMOTIONS})
    return texts, y, (d["split"] == "test").to_numpy()


def learner():
    grid = [{"clf": [LogisticRegression(solver="liblinear", class_weight="balanced", max_iter=2000)], "clf__C": COSTS},
            {"clf": [LinearSVC(class_weight="balanced", max_iter=20000)], "clf__C": COSTS}]  # fmt: skip
    return GridSearchCV(Pipeline([("clf", LogisticRegression())]), grid, scoring="f1",
                        cv=StratifiedKFold(3, shuffle=True, random_state=1), n_jobs=-1)  # fmt: skip


def fit_predict(train_texts, train_y: pd.DataFrame, test_texts):
    """Dictionary, IDF and the six models from the training texts only; predictions for the test texts."""
    fx = EmoTxtFeatures(None, WORDNET, texts=train_texts)
    fx.fit_idf(train_texts)
    Xtr, Xte = fx.transform(train_texts), fx.transform(test_texts)
    models = {e: learner().fit(Xtr, train_y[e].to_numpy()).best_estimator_.named_steps["clf"] for e in EMOTIONS}
    pred = pd.DataFrame({e: models[e].predict(Xte) for e in EMOTIONS})
    return pred, fx, models


def scores(y: pd.DataFrame, pred: pd.DataFrame) -> dict:
    per = {}
    for e in EMOTIONS:
        p, r, f, _ = precision_recall_fscore_support(y[e], pred[e], average="binary", zero_division=0)
        per[e] = {"precision": round(p, 3), "recall": round(r, 3), "f1": round(f, 3), "positives": int(y[e].sum())}
    micro = f1_score(y[EMOTIONS].to_numpy().ravel(), pred[EMOTIONS].to_numpy().ravel(), zero_division=0)
    return {"micro_f1": round(micro, 3), "per_emotion": per}


def training_set(config, so, gh_texts, gh_y, idx):
    parts_t, parts_y = [], []
    if "so" in config:
        parts_t += so[0]
        parts_y.append(so[1])
    if "gh" in config:
        parts_t += [gh_texts[i] for i in idx]
        parts_y.append(gh_y.iloc[idx])
    return parts_t, pd.concat(parts_y, ignore_index=True)


def main():
    so = so_corpus()
    gh_texts, gh_y, is_test = gh_corpus()
    train_idx, test_idx = np.where(~is_test)[0], np.where(is_test)[0]
    print(f"SO {len(so[0])}, GH train {len(train_idx)}, GH test {len(test_idx)}", flush=True)

    results = {}
    for config in ("so", "gh", "so+gh"):
        # 5-fold CV on GH-train (for "so" the GH-train texts are never used for training)
        cv_pred = pd.DataFrame(0, index=range(len(train_idx)), columns=EMOTIONS)
        folds = StratifiedKFold(5, shuffle=True, random_state=0).split(train_idx, gh_y.iloc[train_idx]["anger"])
        for k, (a, b) in enumerate(folds if config != "so" else [(np.array([], int), np.arange(len(train_idx)))]):
            t, y = training_set(config, so, gh_texts, gh_y, train_idx[a])
            pred, _, _ = fit_predict(t, y, [gh_texts[i] for i in train_idx[b]])
            cv_pred.iloc[b] = pred.to_numpy()
        cv = scores(gh_y.iloc[train_idx].reset_index(drop=True), cv_pred)
        # official test, once, with a model trained on the full training data of the configuration
        t, y = training_set(config, so, gh_texts, gh_y, train_idx)
        pred, _, _ = fit_predict(t, y, [gh_texts[i] for i in test_idx])
        test = scores(gh_y.iloc[test_idx].reset_index(drop=True), pred)
        results[config] = {"cv_gh_train": cv, "gh_test": test}
        print(config, "CV micro-F1", cv["micro_f1"], "| test micro-F1", test["micro_f1"],
              {e: test["per_emotion"][e]["f1"] for e in EMOTIONS}, flush=True)  # fmt: skip

    best = max(results, key=lambda c: results[c]["cv_gh_train"]["micro_f1"])
    results["selected"] = best
    print("selected by CV:", best, flush=True)

    # deployed model: the selected configuration on all of its data (SO and/or all 2,000 GH comments)
    t, y = training_set(best, so, gh_texts, gh_y, np.arange(len(gh_texts)))
    _, fx, models = fit_predict(t, y, t[:1])
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir()
    write_dictionary([u for u, _ in fx.unigrams], OUT / "n-grams", "Unigrams")
    write_dictionary([b for b, _ in fx.bigrams], OUT / "n-grams", "Bigrams")
    shutil.copytree(WORDNET, OUT / "WordnetCategories")
    (OUT / "idf.json").write_text(json.dumps(fx.idf))
    np.savez_compressed(OUT / "model.npz", emotions=np.array(EMOTIONS),
                        **{f"w_{e}": m.coef_.ravel().astype(np.float64) for e, m in models.items()},
                        **{f"b_{e}": np.array([float(m.intercept_[0])]) for e, m in models.items()})  # fmt: skip
    results["deployed"] = {"config": best, "training_texts": len(t),
                           "learners": {e: f"{type(m).__name__}(C={m.C})" for e, m in models.items()}}  # fmt: skip
    (OUT / "results.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
