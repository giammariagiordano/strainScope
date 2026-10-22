"""Time-to-abandonment analysis: counting-process dataset, Cox model with time-varying covariates, Kaplan–Meier.

Design
- Subject: developer × project. Time unit: months. Time 0 = the month the subject enters the risk set.
- Entry: the month of the developer's `min_active_months`-th active month (left truncation, so the
  filter never uses information from the future).
- Event: abandonment = last active month τ followed by ≥ K inactive months. Because a gap of K months must
  be observable, events can only be confirmed up to `project_end - K`; everyone still at risk then is censored.
- Covariates for month t summarize the previous W months [t-W, t-1] (lagged, so the activity that defines the
  event month never enters its own covariates).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from sqlalchemy import select
from sqlalchemy.orm import Session

from . import coxdiag
from .config import settings
from .dataset import developer_month_frame
from .models import Developer


@dataclass(frozen=True)
class Covariate:
    key: str
    label: str
    description: str
    count: bool = False  # log1p-transformed when requested
    default: bool = True


COVARIATES = [
    Covariate("core", "Core", "1 if core in any month of the window."),
    Covariate("centrality", "Centrality", "Highest eigenvector centrality in the window."),
    Covariate("off_hours_share", "Off-hours share", "Share of the window's commits on weekends or outside working hours (0 without commits)."),
    Covariate("commits", "Commits", "Commits in the window.", count=True),
    Covariate("comments", "Comments", "Issue/PR and review comments in the window.", count=True),
    Covariate("reviews", "Reviews", "PR reviews submitted in the window.", count=True),
    Covariate("issues_prs_opened", "Issues + PRs opened", "Issues and PRs opened in the window.", count=True, default=False),
    Covariate("active_months", "Active months (window)", "Months with any activity in the window.", default=False),
    Covariate("cum_active_months", "Experience", "Cumulative active months before the interval.", count=True),
    Covariate("truck_factor_member", "Truck-factor member", "Fixed. Measured at the last commit, so it uses future information (look-ahead bias): interpret with care.", default=False),
]  # fmt: skip
COVARIATE_KEYS = [c.key for c in COVARIATES]

CLUSTERS = {
    "person": "Person (GitHub login, across projects)",
    "subject": "Developer × project",
    "project": "Project",
    "none": "None (model-based SEs)",
}
PH_TRANSFORMS = ("rank", "log", "identity")

GROUPINGS = {
    "core": "Core at entry",
    "off_hours_share": "Off-hours share at entry (above vs below median)",
    "truck_factor_member": "Truck-factor member (look-ahead)",
}


def _finite(v) -> float | None:
    """JSON cannot carry inf/NaN (e.g. a diverging hazard ratio under separation)."""
    v = float(v)
    return v if np.isfinite(v) else None


def _midx(month: str) -> int:
    y, m = month.split("-")
    return int(y) * 12 + int(m) - 1


def _rolling(arr: np.ndarray, window: int, how: str) -> np.ndarray:
    """Value for index i summarizes arr[i-window : i] (strictly before i)."""
    out = np.zeros(len(arr))
    for i in range(len(arr)):
        seg = arr[max(0, i - window) : i]
        if len(seg):
            out[i] = seg.sum() if how == "sum" else seg.max()
    return out


def build_intervals(
    db: Session,
    project_ids: list[int] | None = None,
    abandon_months: int | None = None,
    window: int = 3,
    min_active_months: int = 1,
) -> tuple[pd.DataFrame, dict]:
    """Return the counting-process table (one row per subject-month at risk) and exclusion counts."""
    k = abandon_months or settings.abandon_months
    dm = developer_month_frame(db, project_ids, k)
    stats = {"subjects": 0, "late_entry": 0, "no_risk_time": 0, "events": 0}
    if dm.empty:
        return pd.DataFrame(columns=["sid", "person", "project", "developer", "start", "stop", "event", *COVARIATE_KEYS]), stats
    dm = dm.assign(midx=dm["month"].map(_midx))
    logins = dict(
        db.execute(select(Developer.id, Developer.login).where(Developer.id.in_(dm["developer_id"].unique().tolist())))
        .tuples()
        .all()
    )
    project_end = dm.groupby("project_id")["midx"].max()

    rows = []
    for (pid, did), g in dm.groupby(["project_id", "developer_id"], sort=False):
        stats["subjects"] += 1
        g = g.sort_values("midx")
        first, tau = int(g["midx"].iloc[0]), int(g["midx"].iloc[-1])
        obs_end = int(project_end[pid]) - k  # last month in which an abandonment can be confirmed
        if len(g) < min_active_months:
            stats["no_risk_time"] += 1
            continue
        entry = int(g["midx"].iloc[min_active_months - 1])
        if entry >= obs_end:
            stats["late_entry"] += 1
            continue
        event = tau <= obs_end
        stop_m = tau if event else obs_end
        if stop_m <= entry:
            stats["no_risk_time"] += 1  # e.g. one-off contributors: they leave in their entry month
            continue

        # Dense monthly arrays from `first` to `stop_m`.
        n = stop_m - first + 1
        idx = g["midx"].to_numpy() - first
        keep = idx < n
        dense = {}
        for col in ("commits", "comments", "reviews", "off_hours_commits", "centrality", "is_core"):
            a = np.zeros(n)
            a[idx[keep]] = g[col].to_numpy(float)[keep]
            dense[col] = a
        dense["opened"] = np.zeros(n)
        dense["opened"][idx[keep]] = (g["issues_opened"] + g["prs_opened"]).to_numpy(float)[keep]
        active = np.zeros(n)
        active[idx[keep]] = 1

        w = {key: _rolling(dense[key], window, "sum") for key in ("commits", "comments", "reviews", "off_hours_commits", "opened")}
        w_active = _rolling(active, window, "sum")
        w_core = _rolling(dense["is_core"], window, "max")
        w_cent = _rolling(dense["centrality"], window, "max")
        cum_active = np.concatenate([[0], np.cumsum(active)[:-1]])
        tf = int(g["truck_factor_member"].iloc[0])
        project, developer = g["project"].iloc[0], g["developer"].iloc[0]
        login = logins.get(did)
        # Cluster key: the GitHub login links the same person across projects. Internal only: never exported.
        person = f"gh:{login.lower()}" if login else f"{pid}:{did}"

        for t in range(entry + 1, stop_m + 1):
            i = t - first
            c = w["commits"][i]
            rows.append(
                {
                    "sid": f"{pid}:{did}",
                    "person": person,
                    "project": project,
                    "developer": developer,
                    "start": t - entry - 1,
                    "stop": t - entry,
                    "event": int(event and t == stop_m),
                    "core": w_core[i],
                    "centrality": w_cent[i],
                    "off_hours_share": w["off_hours_commits"][i] / c if c else 0.0,
                    "commits": c,
                    "comments": w["comments"][i],
                    "reviews": w["reviews"][i],
                    "issues_prs_opened": w["opened"][i],
                    "active_months": w_active[i],
                    "cum_active_months": cum_active[i],
                    "truck_factor_member": tf,
                }
            )
        stats["events"] += int(event)
    return pd.DataFrame(rows), stats


def fit_cox(
    intervals: pd.DataFrame,
    covariates: list[str],
    log_counts: bool = True,
    standardize: bool = True,
    stratify: bool = True,
    penalizer: float = 0.0,
    cluster: str = "person",
    ph_transform: str = "rank",
) -> dict:
    from lifelines import CoxTimeVaryingFitter

    warnings: list[str] = []
    df = intervals[["sid", "person", "project", "start", "stop", "event", *covariates]].copy()
    transforms = {}
    for c in covariates:
        cov = next(x for x in COVARIATES if x.key == c)
        if log_counts and cov.count:
            df[c] = np.log1p(df[c])
            transforms[c] = "log1p"
        if standardize and df[c].nunique() > 2:
            sd = df[c].std()
            if sd > 0:
                df[c] = (df[c] - df[c].mean()) / sd
                transforms[c] = (transforms.get(c, "") + " z").strip()
    dropped = [c for c in covariates if df[c].nunique() < 2]
    if dropped:
        warnings.append(f"Dropped constant covariates: {', '.join(dropped)}.")
    used = [c for c in covariates if c not in dropped]
    if not used:
        raise ValueError("No covariate varies in this selection")
    events = int(df["event"].sum())
    if events < 10 * len(used):
        warnings.append(
            f"{events} events for {len(used)} covariates (< 10 events per covariate): estimates may be unstable "
            "and overfitted. Use fewer covariates or more projects."
        )
    multi = df["project"].nunique() > 1
    strata = ["project"] if stratify and multi else None
    cols = ["sid", "start", "stop", "event", *used] + (["project"] if strata else [])

    ctv = CoxTimeVaryingFitter(penalizer=penalizer)
    ctv.fit(df[cols], id_col="sid", event_col="event", start_col="start", stop_col="stop", strata=strata)
    beta = ctv.params_[used].to_numpy(float)

    # Residuals, model-based and cluster-robust variance, PH test (Efron ties, see coxdiag).
    parts = coxdiag.efron_parts(
        df[used].to_numpy(float),
        df["start"].to_numpy(float),
        df["stop"].to_numpy(float),
        df["event"].to_numpy(int),
        pd.factorize(df["project"])[0] if strata else np.zeros(len(df), dtype=int),
        beta,
    )
    if penalizer > 0:
        naive = ctv.variance_matrix_.loc[used, used].to_numpy(float)
        warnings.append("Penalized fit: standard errors and p-values are approximate.")
    else:
        try:
            naive = np.linalg.inv(parts.information)
        except np.linalg.LinAlgError:
            naive = ctv.variance_matrix_.loc[used, used].to_numpy(float)
    n_clusters = None
    variance = naive
    if cluster != "none":
        key = {"person": "person", "subject": "sid", "project": "project"}[cluster]
        variance, n_clusters = coxdiag.robust_covariance(naive, parts.score, df[key].to_numpy())
        if n_clusters < 30:
            warnings.append(
                f"Only {n_clusters} clusters: sandwich standard errors are unreliable with few clusters "
                "(they tend to be too small)."
            )
        if cluster == "person":
            shared = df.groupby("person")["project"].nunique()
            if (shared > 1).any():
                warnings.append(
                    f"{int((shared > 1).sum())} people contribute to several projects; their intervals are treated as one cluster."
                )

    naive_se = np.sqrt(np.diag(naive))
    se = np.sqrt(np.diag(variance))
    ph = coxdiag.ph_test(parts.schoenfeld, parts.event_times, naive, ph_transform)
    zcrit = stats.norm.ppf(0.975)

    coefs = []
    for j, c in enumerate(used):
        b, sj = beta[j], se[j]
        z = b / sj if sj > 0 else np.nan
        coefs.append(
            {
                "covariate": c,
                "coef": _finite(b),
                "hr": _finite(np.exp(b)),
                "hr_lower": _finite(np.exp(b - zcrit * sj)),
                "hr_upper": _finite(np.exp(b + zcrit * sj)),
                "se": _finite(sj),
                "naive_se": _finite(naive_se[j]),
                "z": _finite(z),
                "p": _finite(2 * stats.norm.sf(abs(z))) if np.isfinite(z) else None,
                "transform": transforms.get(c, ""),
                "ph": ph["per_covariate"][j],
            }
        )
    separated = [
        c["covariate"] for c in coefs if c["coef"] is None or c["se"] is None or abs(c["coef"]) > 8 or c["se"] > 8
    ]
    if separated:
        warnings.append(
            f"Possible (quasi-)separation for {', '.join(separated)}: the estimate diverges because the covariate "
            "almost perfectly predicts who abandons (or not). Add a penalizer or drop it."
        )
    violating = [c["covariate"] for c in coefs if c["ph"] and c["ph"]["p"] < 0.05]
    if violating:
        warnings.append(
            f"Proportional hazards rejected (p < 0.05) for {', '.join(violating)}: its effect changes over time. "
            "Report it, stratify on it, or model a time interaction on the exported intervals."
        )
    lr = ctv.log_likelihood_ratio_test()
    if multi and not strata:
        warnings.append("Projects pooled without stratification: baseline hazards are assumed equal across projects.")
    return {
        "coefficients": coefs,
        "log_likelihood": float(ctv.log_likelihood_),
        "lr_test": {"statistic": float(lr.test_statistic), "df": int(lr.degrees_freedom), "p": float(lr.p_value)},
        "ph_global": ph["global"],
        "ph_transform": ph_transform,
        "cluster": cluster,
        "n_clusters": n_clusters,
        "events": events,
        "intervals": len(df),
        "subjects": int(df["sid"].nunique()),
        "stratified": bool(strata),
        "warnings": warnings,
    }


def kaplan_meier(intervals: pd.DataFrame, group: str) -> dict:
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test

    if intervals.empty:
        return {"groups": [], "logrank_p": None}
    first = intervals.sort_values("start").groupby("sid").first()
    last = intervals.groupby("sid").agg(duration=("stop", "max"), event=("event", "max"))
    subj = last.join(first[[group]])
    if group == "off_hours_share":
        median = subj[group].median()
        labels = subj[group].gt(median).map({True: f"Above median ({median:.0%})", False: f"At or below median ({median:.0%})"})
    else:
        labels = subj[group].gt(0).map({True: "Yes", False: "No"})
    out = []
    for label in sorted(labels.unique()):
        d = subj[labels == label]
        km = KaplanMeierFitter().fit(d["duration"], d["event"])
        sf = km.survival_function_.iloc[:, 0]
        ci = km.confidence_interval_
        med = km.median_survival_time_
        out.append(
            {
                "label": label,
                "n": len(d),
                "events": int(d["event"].sum()),
                "median": None if not np.isfinite(med) else float(med),
                "curve": [
                    {"t": float(t), "s": float(sf.loc[t]), "lo": float(ci.iloc[i, 0]), "hi": float(ci.iloc[i, 1])}
                    for i, t in enumerate(sf.index)
                ],
            }
        )
    p = None
    if len(out) == 2:
        a, b = (subj[labels == g["label"]] for g in out)
        p = float(logrank_test(a["duration"], b["duration"], a["event"], b["event"]).p_value)
    return {"groups": out, "logrank_p": p}
