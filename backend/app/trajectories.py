"""Event-time trajectories of the psychological signals before disengagement (RQ1).

Design
- Leavers: developers who abandoned a project (no activity for ≥ K months before the end of observation) with at
  least `min_active_months` active months. Their anchor is the last active month.
- Stayers: for each leaver, up to `controls` other developers of the same project who were active in a month M
  with a similar experience (active months before M) and were active again within the K months after M, i.e. who
  had not left at M (as in a survival risk set; they may leave later). M is their pseudo-exit, the anchor. Matching is nearest on experience, without
  replacement, with a fixed seed; M must be at least K months before the end of observation, as for the leavers.
- Rows: the active developer-months in the `horizon` months up to the anchor (relative month -horizon+1 … 0),
  grouped into bins of `bin_size` months.
- Model, per signal: linear mixed model with the bin, the group (leaver vs stayer), their interaction, log1p of the
  month's messages as control, project fixed effects, and a random intercept per subject (developer × project).
  The interaction of the bin closest to the exit tells whether leavers diverge from stayers as the exit approaches.
  p-values are corrected across signals with Benjamini-Hochberg.
"""

import warnings

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control
from sqlalchemy.orm import Session

from .config import settings
from .dataset import developer_month_frame

SIGNALS = {
    "anger_share": "Anger",
    "joy_share": "Joy",
    "love_share": "Love",
    "sadness_share": "Sadness",
    "fear_share": "Fear",
    "surprise_share": "Surprise",
    "negative_emitted_share": "Negativity",
    "we_share": "Belonging (we-talk)",
    "thanks_received_share": "Gratitude received",
    "reactions_pos_per_item": "Positive reactions per item",
}


def _midx(month: str) -> int:
    y, m = month.split("-")
    return int(y) * 12 + int(m) - 1


def event_time_frame(
    db: Session,
    project_ids: list[int] | None = None,
    abandon_months: int | None = None,
    horizon: int = 12,
    bin_size: int = 3,
    min_active_months: int = 3,
    controls: int = 2,
    seed: int = 13,
) -> tuple[pd.DataFrame, dict]:
    """Developer-months of leavers and matched stayers around their (pseudo-)exit, with relative month and bin."""
    k = abandon_months or settings.abandon_months
    dm = developer_month_frame(db, project_ids, k)
    stats = {"leavers": 0, "leavers_matched": 0, "stayers": 0}
    if dm.empty:
        return pd.DataFrame(), stats
    dm = dm.assign(midx=dm["month"].map(_midx), sid=dm["project_id"].astype(str) + ":" + dm["developer_id"].astype(str))
    dm = dm.sort_values(["sid", "midx"])
    dm["experience"] = dm.groupby("sid").cumcount()  # active months before this one
    nxt = dm.groupby("sid")["midx"].shift(-1)
    dm["returns_within_k"] = (nxt - dm["midx"] - 1) < k  # active again within K months
    end = dm.groupby("project_id")["midx"].transform("max")
    totals = dm.groupby("sid")["midx"].transform("size")

    last = dm.groupby("sid")["midx"].transform("max")
    leavers = dm[(dm["midx"] == last) & (end - last >= k) & (totals >= min_active_months)]
    candidates = dm[dm["returns_within_k"] & (end - dm["midx"] >= k) & (dm["experience"] >= min_active_months - 1)]
    stats["leavers"] = len(leavers)

    rng = np.random.default_rng(seed)
    anchors = []  # (sid, anchor month, group, matched set)
    used: set[str] = set()
    for set_id, lv in enumerate(leavers.itertuples()):
        pool = candidates[(candidates["project_id"] == lv.project_id) & (candidates["sid"] != lv.sid) & ~candidates["sid"].isin(used)]
        if pool.empty:
            continue
        # one pseudo-exit per stayer: the month with the experience closest to the leaver's (random tie-break)
        pool = pool.assign(diff=(pool["experience"] - lv.experience).abs(), tie=rng.random(len(pool)))
        best = pool.sort_values(["diff", "tie"]).drop_duplicates("sid").head(controls)
        anchors.append((lv.sid, lv.midx, "leaver", set_id))
        for st in best.itertuples():
            anchors.append((st.sid, st.midx, "stayer", set_id))
            used.add(st.sid)
        stats["leavers_matched"] += 1
        stats["stayers"] += len(best)

    if not anchors:
        return pd.DataFrame(), stats
    a = pd.DataFrame(anchors, columns=["sid", "anchor", "group", "set"])
    rows = dm.merge(a, on="sid")
    rows["rel"] = rows["midx"] - rows["anchor"]
    rows = rows[(rows["rel"] <= 0) & (rows["rel"] > -horizon)].copy()
    rows["bin"] = (-((-rows["rel"]) // bin_size) * bin_size).astype(int)  # 0, -3, -6, -9: start of each bin
    rows["leaver"] = (rows["group"] == "leaver").astype(int)
    rows["log_messages"] = np.log1p(rows["psych_messages"].fillna(0))
    return rows, stats


def trajectories(rows: pd.DataFrame) -> dict:
    """Mean of each signal per group and relative month, with a 95% normal-approximation interval."""
    out = {}
    for s in SIGNALS:
        g = rows.dropna(subset=[s]).groupby(["group", "rel"])[s].agg(["mean", "std", "count"]).reset_index()
        g["ci"] = 1.96 * g["std"] / np.sqrt(g["count"].clip(lower=1))
        out[s] = [
            {"group": r.group, "rel": int(r.rel), "mean": float(r.mean), "ci": None if np.isnan(r.ci) else float(r.ci), "n": int(r.count)}
            for r in g.itertuples()
        ]  # fmt: skip
    return out


def fit_models(rows: pd.DataFrame, min_rows: int = 30) -> list[dict]:
    """Mixed model per signal; reports the leaver × bin interactions (bins relative to the earliest one)."""
    import statsmodels.formula.api as smf

    results = []
    for s, label in SIGNALS.items():
        d = rows.dropna(subset=[s])
        if len(d) < min_rows or d["leaver"].nunique() < 2 or d["bin"].nunique() < 2:
            results.append({"signal": s, "label": label, "error": f"not enough data ({len(d)} developer-months)"})
            continue
        earliest = int(d["bin"].min())
        fe = " + C(project)" if d["project"].nunique() > 1 else ""
        formula = f"{s} ~ C(bin, Treatment({earliest})) * leaver + log_messages{fe}"
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = smf.mixedlm(formula, d, groups=d["sid"]).fit(reml=True, method=["lbfgs", "powell"])
        except Exception as e:  # singular fits on tiny samples are reported, not raised
            results.append({"signal": s, "label": label, "error": str(e)[:200]})
            continue
        ci = fit.conf_int()
        terms = []
        for name in fit.params.index:
            if name.endswith(":leaver") and name.startswith("C(bin"):
                b = int(name.split("[T.")[1].split("]")[0])
                terms.append({"bin": b, "coef": float(fit.params[name]), "lower": float(ci.loc[name, 0]),
                              "upper": float(ci.loc[name, 1]), "p": float(fit.pvalues[name])})  # fmt: skip
        terms.sort(key=lambda t: t["bin"])
        base = "leaver"
        results.append({
            "signal": s, "label": label, "n_rows": int(len(d)), "n_subjects": int(d["sid"].nunique()),
            "baseline_bin": earliest,
            "leaver_at_baseline": {"coef": float(fit.params[base]), "p": float(fit.pvalues[base])},
            "interactions": terms,
        })  # fmt: skip
    # Benjamini-Hochberg over the interaction of the bin closest to the exit, one test per signal
    closest = [(i, r["interactions"][-1]["p"]) for i, r in enumerate(results) if r.get("interactions")]
    if closest:
        q = false_discovery_control([p for _, p in closest])
        for (i, _), qi in zip(closest, q):
            results[i]["interactions"][-1]["q"] = float(qi)
    return results
