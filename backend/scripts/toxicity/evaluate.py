"""Shared evaluation for conversation-level toxicity classifiers (docs/TOXICITY.md).

Metrics on the toxic class: precision, recall, F1 at a threshold chosen on the validation split (max F1 at the
natural toxicity rate), plus threshold-free average precision (PR-AUC), which is the meaningful summary when the
positive class is below 1%. 95% confidence intervals by bootstrap over repositories (the unit of sampling).
"""

import json

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, precision_score, recall_score


def best_threshold(y, scores) -> float:
    p, r, t = precision_recall_curve(y, scores)
    f1 = 2 * p * r / np.clip(p + r, 1e-12, None)
    return float(t[np.argmax(f1[:-1])])


def metrics(y, scores, threshold) -> dict:
    y = np.asarray(y)
    pred = (np.asarray(scores) >= threshold).astype(int)
    return {
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "average_precision": float(average_precision_score(y, scores)),
        "flagged": int(pred.sum()),
        "positives": int(y.sum()),
        "n": int(len(y)),
    }


def bootstrap_by_group(y, scores, groups, threshold, n=1000, seed=0) -> dict:
    """Resample repositories with replacement; returns 95% intervals for P, R, F1 and AP."""
    y, scores, groups = np.asarray(y), np.asarray(scores), np.asarray(groups)
    uniq = np.unique(groups)
    index = {g: np.flatnonzero(groups == g) for g in uniq}
    rng = np.random.default_rng(seed)
    out = {k: [] for k in ("precision", "recall", "f1", "average_precision")}
    for _ in range(n):
        idx = np.concatenate([index[g] for g in rng.choice(uniq, len(uniq))])
        if y[idx].sum() == 0:
            continue
        m = metrics(y[idx], scores[idx], threshold)
        for k in out:
            out[k].append(m[k])
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in out.items()}


def report(name, val, test, imran=None, out_path=None) -> dict:
    """val/test/imran: (y, scores[, groups]). Threshold from validation, applied unchanged elsewhere."""
    thr = best_threshold(val[0], val[1])
    res = {"model": name, "threshold": thr, "val": metrics(val[0], val[1], thr), "test": metrics(test[0], test[1], thr)}
    if len(test) > 2:
        res["test_ci95"] = bootstrap_by_group(test[0], test[1], test[2], thr)
    if imran is not None:
        res["imran"] = metrics(imran[0], imran[1], thr)
    t, ci = res["test"], res.get("test_ci95", {})
    fmt = lambda k: f"{t[k]:.3f}" + (f" [{ci[k][0]:.2f}-{ci[k][1]:.2f}]" if k in ci else "")  # noqa: E731
    print(f"{name:40} test P {fmt('precision')} R {fmt('recall')} F1 {fmt('f1')} AP {fmt('average_precision')}"
          f" | flagged {t['flagged']}/{t['n']} (true {t['positives']})"
          + (f" | Imran F1 {res['imran']['f1']:.3f} AP {res['imran']['average_precision']:.3f}" if imran is not None else ""))
    if out_path:
        with open(out_path, "w") as f:
            json.dump(res, f, indent=2)
    return res
