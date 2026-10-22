"""Cox model diagnostics for counting-process data with Efron ties.

Residuals follow the definitions used by R's `survival` package:
- score residuals (Lin & Wei 1989) for cluster-robust "sandwich" variance, as in `coxph(..., cluster = id)`;
- Schoenfeld residuals for the Grambsch–Therneau (1994) test of proportional hazards, in its classic form
  (the per-covariate and global statistics of `cox.zph` before survival 3.0, and of lifelines'
  `proportional_hazard_test`).

A row (start, stop] is at risk at event time t when start < t <= stop.
"""

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class EfronParts:
    score: np.ndarray  # n × p score residuals (sum to the score vector, ≈ 0 at the MLE)
    information: np.ndarray  # p × p observed information
    schoenfeld: np.ndarray  # d × p Schoenfeld residuals, one row per event
    event_times: np.ndarray  # d event times, aligned with `schoenfeld`


def efron_parts(
    X: np.ndarray, start: np.ndarray, stop: np.ndarray, event: np.ndarray, strata: np.ndarray, beta: np.ndarray
) -> EfronParts:
    n, p = X.shape
    w = np.exp(X @ beta)
    score = np.zeros((n, p))
    info = np.zeros((p, p))
    sch_rows, sch_times = [], []
    for s in np.unique(strata):
        in_s = strata == s
        for t in np.unique(stop[in_s & (event == 1)]):
            risk = in_s & (start < t) & (stop >= t)
            dead = risk & (stop == t) & (event == 1)
            d = int(dead.sum())
            Xr, wr = X[risk], w[risk]
            Xd, wd = X[dead], w[dead]
            S0, S1, S2 = wr.sum(), wr @ Xr, (Xr * wr[:, None]).T @ Xr
            D0, D1, D2 = wd.sum(), wd @ Xd, (Xd * wd[:, None]).T @ Xd
            means, dens = [], []
            for k in range(d):
                f = k / d
                den = S0 - f * D0
                mean = (S1 - f * D1) / den
                info += (S2 - f * D2) / den - np.outer(mean, mean)
                means.append(mean)
                dens.append(den)
            means, dens = np.array(means), np.array(dens)
            avg_mean = means.mean(0)

            # Events: Schoenfeld residual, also the "observed" part of the score residual.
            for i in np.flatnonzero(dead):
                sch_rows.append(X[i] - avg_mean)
                sch_times.append(t)
                score[i] += X[i] - avg_mean

            # Compensator: every row at risk, deaths with Efron's down-weighting (1 - k/d).
            idx = np.flatnonzero(risk)
            is_dead = dead[idx]
            for k in range(d):
                f = k / d
                weight = w[idx] * np.where(is_dead, 1 - f, 1.0) / dens[k]
                score[idx] -= weight[:, None] * (X[idx] - means[k])
    return EfronParts(score, info, np.array(sch_rows).reshape(-1, p), np.array(sch_times))


def robust_covariance(variance: np.ndarray, score: np.ndarray, clusters: np.ndarray) -> tuple[np.ndarray, int]:
    """Sandwich V B V with B = Σ_c U_c U_cᵀ, U_c the summed score residuals of cluster c (no small-sample factor)."""
    _, inverse = np.unique(clusters, return_inverse=True)
    U = np.zeros((inverse.max() + 1, score.shape[1]))
    np.add.at(U, inverse, score)
    return variance @ (U.T @ U) @ variance, U.shape[0]


def ph_test(schoenfeld: np.ndarray, event_times: np.ndarray, variance: np.ndarray, transform: str = "rank") -> dict:
    """Grambsch–Therneau test: correlation between scaled Schoenfeld residuals and g(time).

    Per covariate j: T_j = d (a V)_j² / (V_jj Σ g̃²); global: T = d a V aᵀ / Σ g̃², with a = g̃ᵀ S,
    S the Schoenfeld residuals, g̃ the centred transformed event times and d the number of events.
    """
    d, p = schoenfeld.shape
    g = stats.rankdata(event_times) if transform == "rank" else np.log(event_times) if transform == "log" else event_times
    g = g - g.mean()
    gg = float(g @ g)
    if d < 3 or gg == 0:
        return {"per_covariate": [None] * p, "global": None}
    a = g @ schoenfeld
    aV = a @ variance
    per = []
    for j in range(p):
        T = d * aV[j] ** 2 / (variance[j, j] * gg)
        per.append({"statistic": float(T), "df": 1, "p": float(stats.chi2.sf(T, 1))})
    T = float(d * a @ variance @ a / gg)
    return {"per_covariate": per, "global": {"statistic": T, "df": p, "p": float(stats.chi2.sf(T, p))}}
