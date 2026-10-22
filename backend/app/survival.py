"""Time-to-disengagement analysis: counting-process dataset, Cox model with time-varying covariates, Kaplan–Meier.

Design
- Subject: developer × project. Time unit: calendar months since entry.
- Entry: the developer's `min_active_months`-th active month (left truncation, so the filter never uses
  information from the future). The entry month itself is not at risk.
- Risk set: only the developer's active months. "At risk" means: after this active month, will they stop?
  During a pause they are not at risk (they cannot start a pause they are already in), and they re-enter
  when they come back. Keeping inactive months at risk would teach the model that inactivity protects
  from leaving, since those months can never be event months.
- Break: an active month followed by ≥ K inactive months (returning later or not).
- Outcomes:
  * "abandonment" (default): only the final exit is an event — the developer never comes back. Pauses
    followed by a return are never events, however long;
  * "break": every pause of ≥ K months is an event, with or without return — recurrent events
    (Andersen–Gill), so standard errors must be clustered by subject or person.
- K is needed for abandonment only because the future is unknown: a final silence shorter than K months
  before the end of the data may still end with a return, so it is censored, not counted.
- A break can only be confirmed when K inactive months are observable, so active months later than
  `project_end - K` are censored.
- Covariates for month t summarize the previous W calendar months [t-W, t-1] (lagged).
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
    Covariate("gap_before", "Months since previous activity", "Inactive months between the previous active month and this one (0 if consecutive).", count=True, default=False),
    Covariate("toxic_emitted_share", "Toxicity emitted", "Share of the window's messages classified as toxic (ToxiCR); 0 without messages.", default=False),
    Covariate("toxic_received", "Toxic messages received", "Toxic messages by others on the developer's issues/PRs in the window (ToxiCR).", count=True, default=False),
    Covariate("negative_emitted_share", "Negativity emitted", "Share of the window's messages with negative polarity (Senti4SD); 0 without messages.", default=False),
    Covariate("negative_received", "Negative messages received", "Negative messages by others on the developer's issues/PRs in the window (Senti4SD).", count=True, default=False),
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
OUTCOMES = {
    "abandonment": "Abandonment: stops and never comes back",
    "break": "Any long pause (≥ K months, with or without return; recurrent)",
}

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


def _share(part: float, total: float) -> float:
    if np.isnan(part) or np.isnan(total):
        return np.nan
    return part / total if total else 0.0


def _rolling(arr: np.ndarray, window: int, how: str) -> np.ndarray:
    """Value for index i summarizes arr[i-window : i] (strictly before i)."""
    out = np.zeros(len(arr))
    for i in range(len(arr)):
        seg = arr[max(0, i - window) : i]
        if len(seg):
            out[i] = seg.sum() if how == "sum" else seg.max()
    return out


def subject_rows(
    months: np.ndarray,
    values: dict[str, np.ndarray],
    truck_factor_member: int,
    project_end: int,
    k: int,
    window: int,
    min_active_months: int,
    outcome: str,
) -> tuple[list[dict], str]:
    """Risk-set rows for one subject. `months` are the sorted active month indices; `values` hold the
    monthly activity aligned with them. Returns (rows, status) with status ok | no_risk_time | late_entry."""
    if len(months) < min_active_months:
        return [], "no_risk_time"
    first, last = int(months[0]), int(months[-1])
    entry = int(months[min_active_months - 1])
    obs_end = project_end - k  # latest month whose following K months are observable
    if entry >= obs_end:
        return [], "late_entry"

    # Dense calendar arrays from the first to the last active month, for the lagged windows.
    n = last - first + 1
    idx = months - first
    dense = {}
    for col in ("commits", "comments", "reviews", "off_hours_commits", "centrality", "is_core", "opened",
                "messages", "toxic_emitted", "toxic_received", "negative_emitted", "negative_received"):
        a = np.zeros(n)
        a[idx] = values.get(col, np.full(len(months), np.nan))  # NaN = not measured (tool not run)
        dense[col] = a
    for col in ("messages", "toxic_emitted", "toxic_received", "negative_emitted", "negative_received"):
        if np.isnan(dense[col]).all():
            dense[col][:] = np.nan  # keep "not measured" for inactive months too
        else:
            dense[col] = np.nan_to_num(dense[col])
    active = np.zeros(n)
    active[idx] = 1
    w = {key: _rolling(dense[key], window, "sum") for key in ("commits", "comments", "reviews", "off_hours_commits", "opened",
                                                               "messages", "toxic_emitted", "toxic_received",
                                                               "negative_emitted", "negative_received")}  # fmt: skip
    w_active = _rolling(active, window, "sum")
    w_core = _rolling(dense["is_core"], window, "max")
    w_cent = _rolling(dense["centrality"], window, "max")
    cum_active = np.concatenate([[0], np.cumsum(active)[:-1]])

    rows, episode = [], 0
    for j, t in enumerate(months):
        t = int(t)
        if t <= entry or t > obs_end:
            continue
        nxt = int(months[j + 1]) if j + 1 < len(months) else None
        gap_after = (nxt - t - 1) if nxt is not None else (project_end - t)
        is_break = gap_after >= k
        returned = nxt is not None
        event = is_break if outcome == "break" else (is_break and not returned)
        i = t - first
        c = w["commits"][i]
        rows.append(
            {
                "episode": episode,
                "start": t - entry - 1,
                "stop": t - entry,
                "event": int(event),
                "event_type": ("break_return" if returned else "abandonment") if is_break else "",
                "core": w_core[i],
                "centrality": w_cent[i],
                "off_hours_share": w["off_hours_commits"][i] / c if c else 0.0,
                "commits": c,
                "comments": w["comments"][i],
                "reviews": w["reviews"][i],
                "issues_prs_opened": w["opened"][i],
                "active_months": w_active[i],
                "cum_active_months": cum_active[i],
                "gap_before": t - int(months[j - 1]) - 1 if j > 0 else 0,
                "toxic_emitted_share": _share(w["toxic_emitted"][i], w["messages"][i]),
                "toxic_received": w["toxic_received"][i],
                "negative_emitted_share": _share(w["negative_emitted"][i], w["messages"][i]),
                "negative_received": w["negative_received"][i],
                "truck_factor_member": truck_factor_member,
            }
        )
        if is_break and outcome == "break":
            episode += 1
    return rows, ("ok" if rows else "no_risk_time")


def build_intervals(
    db: Session,
    project_ids: list[int] | None = None,
    abandon_months: int | None = None,
    window: int = 3,
    min_active_months: int = 1,
    outcome: str = "abandonment",
) -> tuple[pd.DataFrame, dict]:
    """Return the counting-process table (one row per subject × active month at risk) and summary counts."""
    k = abandon_months or settings.abandon_months
    dm = developer_month_frame(db, project_ids, k)
    stats = {"subjects": 0, "late_entry": 0, "no_risk_time": 0, "events": 0, "subjects_with_event": 0,
             "breaks_with_return": 0, "abandonments": 0}  # fmt: skip
    columns = ["sid", "person", "project", "developer", "episode", "start", "stop", "event", "event_type", *COVARIATE_KEYS]
    if dm.empty:
        return pd.DataFrame(columns=columns), stats
    dm = dm.assign(midx=dm["month"].map(_midx), opened=dm["issues_opened"] + dm["prs_opened"])
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
        values = {c: g[c].to_numpy(float) for c in ("commits", "comments", "reviews", "off_hours_commits", "centrality",
                                                     "is_core", "opened", "messages", "toxic_emitted", "toxic_received",
                                                     "negative_emitted", "negative_received")}  # fmt: skip
        sub, status = subject_rows(
            g["midx"].to_numpy(), values, int(g["truck_factor_member"].iloc[0]), int(project_end[pid]),
            k, window, min_active_months, outcome,
        )  # fmt: skip
        if status != "ok":
            stats[status] += 1
            continue
        login = logins.get(did)
        # Cluster key: the GitHub login links the same person across projects. Internal only: never exported.
        person = f"gh:{login.lower()}" if login else f"{pid}:{did}"
        ids = {"sid": f"{pid}:{did}", "person": person, "project": g["project"].iloc[0], "developer": g["developer"].iloc[0]}
        rows.extend({**ids, **r} for r in sub)
        events = sum(r["event"] for r in sub)
        stats["events"] += events
        stats["subjects_with_event"] += int(events > 0)
        stats["breaks_with_return"] += sum(r["event_type"] == "break_return" for r in sub)
        stats["abandonments"] += sum(r["event_type"] == "abandonment" for r in sub)
    return pd.DataFrame(rows, columns=columns), stats


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
    df = intervals[["sid", "person", "project", "episode", "start", "stop", "event", *covariates]].copy()
    unmeasured = df[covariates].isna().any(axis=1)
    if unmeasured.any():
        skipped = sorted(df.loc[unmeasured, "project"].unique())
        warnings.append(
            f"{len(skipped)} project(s) excluded because a selected covariate was not measured there "
            f"(text classification not run): {', '.join(skipped[:5])}{'…' if len(skipped) > 5 else ''}."
        )
        df = df[~unmeasured]
    # One id per episode: with recurrent events lifelines expects one event per id. The partial
    # likelihood does not depend on the id; dependence between episodes is handled by clustering.
    df["eid"] = df["sid"] + ":" + df["episode"].astype(str)
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
    cols = ["eid", "start", "stop", "event", *used] + (["project"] if strata else [])

    ctv = CoxTimeVaryingFitter(penalizer=penalizer)
    ctv.fit(df[cols], id_col="eid", event_col="event", start_col="start", stop_col="stop", strata=strata)
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

    recurrent = bool((df.groupby("sid")["event"].sum() > 1).any())
    if recurrent and cluster in ("none", "project"):
        warnings.append(
            "Recurrent events (several breaks per developer): cluster by person or developer × project, "
            "otherwise standard errors ignore the dependence between a developer's breaks."
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
        "recurrent": recurrent,
        "warnings": warnings,
    }


def kaplan_meier(intervals: pd.DataFrame, group: str) -> dict:
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test

    if intervals.empty:
        return {"groups": [], "logrank_p": None}
    # Time to the FIRST event: KM assumes one event per subject.
    iv = intervals.sort_values(["sid", "stop"])
    first = iv.groupby("sid").first()
    first_event = iv[iv["event"] == 1].groupby("sid")["stop"].min()
    last_stop = iv.groupby("sid")["stop"].max()
    subj = pd.DataFrame({"duration": first_event.reindex(last_stop.index).fillna(last_stop), "event": last_stop.index.isin(first_event.index).astype(int)})
    subj = subj.join(first[[group]])
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
