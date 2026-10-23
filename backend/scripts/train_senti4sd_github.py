"""Retrain Senti4SD's learner on the GitHub gold standard (Novielli et al., MSR 2020) and save it as
external/pySenti4SD/Senti4SD-github.model. Run with external/senti4sd-venv/bin/python from external/.

Inputs: senti4sd-work/gh_gold.csv (ID;Polarity;Text) and senti4sd-work/features/gh_gold.csv, the features
extracted by Senti4SD-fast.jar (see README, "Text classification"). Also prints the 10-fold macro-F1.
"""

import csv
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.multiclass import OneVsRestClassifier

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app" / "tools"))
from senti4sd_model import CLASSES, LiblinearModel  # noqa: E402


def learner():  # LIBLINEAR solver 6 (L1-regularized LR), C=0.5, one-vs-rest, bias 1, as in Senti4SD_info
    return OneVsRestClassifier(LogisticRegression(penalty="l1", C=0.5, solver="liblinear", intercept_scaling=1, max_iter=1000))


X = pd.read_csv("senti4sd-work/features/gh_gold.csv").drop(columns=["id"]).to_numpy(np.float32)
y = np.array([r[1] for r in list(csv.reader(open("senti4sd-work/gh_gold.csv", encoding="utf-8"), delimiter=";"))[1:]])
cv = cross_val_predict(learner(), X, y, cv=StratifiedKFold(10, shuffle=True, random_state=0), n_jobs=4)
print(f"10-fold macro-F1 on GitHub gold: {f1_score(y, cv, average='macro'):.3f}")

clf = learner().fit(X, y)
label_ids = [int(np.where(CLASSES == c)[0][0]) for c in clf.classes_]
W = np.vstack([np.column_stack([e.coef_[0] for e in clf.estimators_]), [e.intercept_[0] for e in clf.estimators_]])
out = Path("pySenti4SD/Senti4SD-github.model")
with open(out, "w") as f:
    f.write("solver_type L1R_LR\nnr_class 3\n" + "label " + " ".join(map(str, label_ids)) + "\n")
    f.write(f"nr_feature {X.shape[1]}\nbias 1\nw\n")
    for row in W:
        f.write(" ".join(repr(float(v)) for v in row) + " \n")
assert (LiblinearModel(out).predict(X) == clf.predict(X)).all()
print(f"saved {out}")
