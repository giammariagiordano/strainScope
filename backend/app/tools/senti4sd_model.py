"""Pure-numpy prediction for Senti4SD's LIBLINEAR model (no native liblinear needed).

Senti4SD.model is a LIBLINEAR text model (L1-regularized logistic regression, one-vs-rest, 3 classes, bias 1).
LIBLINEAR predicts argmax_k (w_k · x + w_k,bias · bias) and ignores features whose index exceeds nr_feature;
this module does exactly the same. Verified to reproduce the published test accuracy (see README).
"""

import numpy as np

# pySenti4SD decodes LIBLINEAR's integer labels with sklearn's LabelEncoder fitted on these three strings.
CLASSES = np.array(sorted(["positive", "negative", "neutral"]))  # 0 negative, 1 neutral, 2 positive


class LiblinearModel:
    def __init__(self, path):
        header, weights = {}, []
        with open(path) as f:
            for line in f:
                key, *vals = line.split()
                if key == "w":
                    break
                header[key] = vals
            for line in f:
                weights.append([float(v) for v in line.split()])
        self.labels = np.array([int(v) for v in header["label"]])
        self.nr_feature = int(header["nr_feature"][0])
        self.bias = float(header["bias"][0])
        w = np.array(weights)
        self.w, self.w_bias = w[: self.nr_feature], w[self.nr_feature]

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = X[:, : self.nr_feature]
        scores = X @ self.w + self.bias * self.w_bias
        return CLASSES[self.labels[scores.argmax(axis=1)]]
