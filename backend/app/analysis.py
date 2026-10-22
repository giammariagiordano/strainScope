"""Correlation analysis between developer-level variables."""

import math

import numpy as np
import pandas as pd
from scipy import stats

METHODS = ("spearman", "pearson")


def _transform(s: pd.Series, method: str) -> pd.Series:
    return s.rank() if method == "spearman" else s


def _residualize(y: np.ndarray, controls: np.ndarray) -> np.ndarray:
    X = np.column_stack([np.ones(len(y)), controls])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


def correlate_pair(df: pd.DataFrame, x: str, y: str, method: str, controls: list[str]) -> dict:
    """(Partial) correlation with listwise deletion on x, y and the controls.

    Partial correlation: both variables are transformed (ranked for Spearman), regressed on the
    transformed controls, and the residuals are correlated; the t-test uses n - 2 - k degrees of freedom.
    """
    cols = list(dict.fromkeys([x, y, *controls]))
    d = df[cols].dropna()
    n, k = len(d), len(controls)
    if n < k + 4:
        return {"r": None, "p": None, "n": n}
    t = {c: _transform(d[c], method).to_numpy(float) for c in cols}
    if np.std(t[x]) == 0 or np.std(t[y]) == 0:
        return {"r": None, "p": None, "n": n}
    if k:
        C = np.column_stack([t[c] for c in controls])
        a, b = _residualize(t[x], C), _residualize(t[y], C)
    else:
        a, b = t[x], t[y]
    if np.std(a) == 0 or np.std(b) == 0:
        return {"r": None, "p": None, "n": n}
    r = float(np.corrcoef(a, b)[0, 1])
    dof = n - 2 - k
    if abs(r) >= 1:
        p = 0.0
    else:
        tstat = r * math.sqrt(dof / (1 - r * r))
        p = float(2 * stats.t.sf(abs(tstat), dof))
    return {"r": r, "p": p, "n": n}


def benjamini_hochberg(pvalues: list[float]) -> list[float]:
    m = len(pvalues)
    if m == 0:
        return []
    order = np.argsort(pvalues)
    ranked = np.array(pvalues)[order] * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1].clip(max=1)
    out = np.empty(m)
    out[order] = q
    return out.tolist()


def correlation_matrix(df: pd.DataFrame, variables: list[str], method: str, controls: list[str]) -> dict:
    variables = [v for v in variables if v not in controls]
    cells: dict[tuple[str, str], dict] = {}
    pairs = []
    for i, x in enumerate(variables):
        for y in variables[i + 1 :]:
            res = correlate_pair(df, x, y, method, controls)
            cells[(x, y)] = cells[(y, x)] = res
            if res["r"] is not None:
                pairs.append({"x": x, "y": y, **res})
    for pair, q in zip(pairs, benjamini_hochberg([p["p"] for p in pairs])):
        pair["q"] = q
        cells[(pair["x"], pair["y"])]["q"] = cells[(pair["y"], pair["x"])]["q"] = q
    matrix = [
        [{"r": 1.0, "p": None, "q": None, "n": int(df[x].notna().sum())} if x == y else cells[(x, y)] for y in variables]
        for x in variables
    ]
    pairs.sort(key=lambda p: -abs(p["r"]))
    return {"variables": variables, "matrix": matrix, "pairs": pairs}
