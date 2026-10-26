"""Risk-set and event definitions on hand-made timelines (months are integer indices)."""

import numpy as np

from app.survival import subject_rows

K, WINDOW = 12, 3


def timeline(months, end, outcome="break", min_active=1):
    months = np.array(months)
    ones = np.ones(len(months))
    values = {c: ones.copy() for c in ("commits", "comments", "reviews", "centrality", "opened")}
    values |= {"off_hours_commits": np.zeros(len(months)), "is_core": np.zeros(len(months))}
    values |= {c: ones.copy() for c in ("messages", "toxic_emitted", "toxic_received", "negative_emitted", "negative_received")}
    return subject_rows(months, values, 0, end, K, WINDOW, min_active, outcome)


def at_risk(rows):
    return [r["stop"] for r in rows]  # months since entry


def test_one_year_pause_then_return_is_a_break_not_an_abandonment():
    # Active months 0-3, silent for 13 months, back at 17-20; data ends at 60.
    rows, status = timeline([0, 1, 2, 3, 17, 18, 19, 20], end=60)
    assert status == "ok"
    # Only active months after entry are at risk; the pause (4-16) never appears.
    assert at_risk(rows) == [1, 2, 3, 17, 18, 19, 20]
    events = {r["stop"]: r["event_type"] for r in rows if r["event"]}
    assert events == {3: "break_return", 20: "abandonment"}
    # A new episode starts after the break (recurrent events).
    assert [r["episode"] for r in rows] == [0, 0, 0, 1, 1, 1, 1]
    # The month after the pause knows how long the developer was away.
    assert rows[3]["gap_before"] == 13


def test_abandonment_outcome_ignores_breaks_with_return():
    rows, _ = timeline([0, 1, 2, 3, 17, 18, 19, 20], end=60, outcome="abandonment")
    assert at_risk(rows) == [1, 2, 3, 17, 18, 19, 20]  # same risk set
    assert [r["stop"] for r in rows if r["event"]] == [20]
    assert {r["episode"] for r in rows} == {0}


def test_pause_shorter_than_threshold_is_not_an_event():
    rows, _ = timeline([0, 1, 12, 13], end=60)  # 10 silent months
    assert [r["stop"] for r in rows if r["event"]] == [13]  # only the final exit


def test_months_too_close_to_the_end_are_censored():
    # Active until month 55, data ends at 60: a 12-month gap cannot be observed yet.
    rows, _ = timeline([40, 41, 50, 55], end=60)
    assert at_risk(rows) == [1]  # month 41 only; 50 and 55 are after end - K = 48
    assert not any(r["event"] for r in rows)


def test_one_off_contributor_has_no_time_at_risk():
    rows, status = timeline([5], end=60)
    assert rows == [] and status == "no_risk_time"


def test_delayed_entry_uses_the_nth_active_month():
    rows, _ = timeline([0, 1, 2, 3, 4], end=60, min_active=3)  # entry at month 2
    assert at_risk(rows) == [1, 2]  # months 3 and 4


def test_late_entry():
    rows, status = timeline([50, 51], end=60)
    assert rows == [] and status == "late_entry"


def test_covariates_are_lagged():
    rows, _ = timeline([0, 1, 2, 3, 17], end=60)
    first_after_pause = next(r for r in rows if r["stop"] == 17)
    # Window = months 14-16, all silent: nothing from month 17 itself leaks in.
    assert first_after_pause["commits"] == 0 and first_after_pause["active_months"] == 0
    month3 = next(r for r in rows if r["stop"] == 3)
    assert month3["commits"] == 3 and month3["active_months"] == 3  # months 0-2


def test_strain_covariates_use_the_lagged_window():
    months = np.array([0, 1, 2, 3])
    ones, zeros = np.ones(4), np.zeros(4)
    values = {c: ones.copy() for c in ("commits", "comments", "reviews", "centrality", "opened")}
    values |= {"off_hours_commits": zeros, "is_core": zeros, "negative_emitted": zeros, "negative_received": zeros,
               "messages": np.array([4.0, 4, 4, 4]), "toxic_emitted": np.array([0.0, 2, 0, 0]),
               "toxic_received": np.array([1.0, 0, 0, 3])}
    rows, _ = subject_rows(months, values, 0, 60, K, WINDOW, 1, "abandonment")
    month3 = next(r for r in rows if r["stop"] == 3)
    assert month3["toxic_emitted_share"] == 2 / 12  # months 0-2: 2 toxic of 12 messages
    assert month3["toxic_received"] == 1  # month 3's own 3 toxic replies are not in its window


def test_unmeasured_text_metrics_stay_missing():
    rows, _ = subject_rows(np.array([0, 1, 2]), {c: np.ones(3) for c in ("commits", "comments", "reviews", "centrality", "opened", "off_hours_commits", "is_core")},
                           0, 60, K, WINDOW, 1, "abandonment")  # fmt: skip
    assert all(np.isnan(r["toxic_emitted_share"]) and np.isnan(r["toxic_received"]) for r in rows)


def test_community_toxicity_uses_previous_window_of_the_project():
    from app.survival import _community_share
    community = {10: (4, 1), 11: (6, 0), 12: (10, 10)}  # month -> (classified conversations, toxic)
    assert _community_share(community, 12, 3) == 1 / 10  # months 9-11 only; month 12 itself excluded
    assert _community_share(community, 30, 3) == 0.0  # no conversations in the window
    assert np.isnan(_community_share(None, 12, 3))  # not measured


# -- context covariates: project decline, time-varying truck factor, responses ------------------------------

from app.survival import ProjectSeries, context_covariates  # noqa: E402


def series(tf_history=((10, {1}), (20, {2}))):
    activity = {m: {1: 1.0, 2: 2.0} for m in range(0, 12)} | {m: {1: 1.0, 3: 1.0} for m in range(12, 30)}
    issues = {1: {14: [(2.0, False), (30.0, True)], 25: [(4.0, False)]}}
    prs = {1: {13: [(True, 13), (False, 14), (False, 15), (True, None)]}}  # (merged, month resolved)
    return ProjectSeries(activity, list(tf_history) if tf_history is not None else None, issues, prs)


def test_project_context_excludes_the_developer_and_is_lagged():
    c = context_covariates(series(), did=1, t=15, window=3)
    assert c["project_active_devs"] == 1  # only developer 3 in months 12-14; developer 1 is the subject
    # others' activity: months 12-14 = 3 (dev 3), months 9-11 = 6 (dev 2)
    assert np.isclose(c["project_trend"], np.log1p(3) - np.log1p(6))
    assert context_covariates(series(), did=1, t=12, window=3)["project_active_devs"] == 1  # months 9-11: dev 2


def test_truck_factor_uses_the_snapshot_before_the_interval_month():
    assert context_covariates(series(), 1, t=10, window=3)["truck_factor_member_tv"] == 0  # snapshot of month 10 not yet known
    assert context_covariates(series(), 1, t=11, window=3)["truck_factor_member_tv"] == 1
    assert context_covariates(series(), 1, t=21, window=3)["truck_factor_member_tv"] == 0  # replaced by dev 2
    assert np.isnan(context_covariates(series(None), 1, t=21, window=3)["truck_factor_member_tv"])  # not computed


def test_responses_cover_issues_opened_in_the_window():
    c = context_covariates(series(), 1, t=15, window=3)
    assert c["unanswered_share"] == 0.5 and c["response_days"] == 16.0  # median of 2 and 30 (capped)
    none = context_covariates(series(), 1, t=20, window=3)  # nothing opened in months 17-19
    assert none["unanswered_share"] == 0 and np.isnan(none["response_days"])


def test_pr_rejections_count_only_outcomes_known_before_the_interval_month():
    assert context_covariates(series(), 1, t=15, window=3)["pr_rejected_share"] == 0.5  # resolved in 13, 14
    assert np.isclose(context_covariates(series(), 1, t=16, window=3)["pr_rejected_share"], 2 / 3)  # + month 15
    assert context_covariates(series(), 1, t=13, window=3)["pr_rejected_share"] == 0  # opened in 13: not in window


def test_disengagement_counts_only_exits_without_activity_elsewhere():
    months = np.array([0, 1, 2, 3])
    values = {c: np.ones(4) for c in ("commits", "comments", "reviews", "centrality", "opened")}
    values |= {"off_hours_commits": np.zeros(4), "is_core": np.zeros(4)}
    run = lambda left: subject_rows(months, values, 0, 60, K, WINDOW, 1, "disengagement", left_github=left)  # noqa: E731
    rows, _ = run(True)
    assert [r["stop"] for r in rows if r["event"]] == [3]
    rows, _ = run(False)  # moved to other repositories: censored at the exit
    assert not any(r["event"] for r in rows) and rows[-1]["event_type"] == "moved_elsewhere"
    assert run(None) == ([], "unknown_after_exit")
    rows, _ = subject_rows(months, values, 0, 60, K, WINDOW, 1, "abandonment", left_github=False)
    assert [r["stop"] for r in rows if r["event"]] == [3]  # still an abandonment of the project
