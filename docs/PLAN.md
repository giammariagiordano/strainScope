# Implementation plan

Goal: a web platform that takes one or more GitHub repository URLs, runs a resumable mining pipeline,
and computes per-developer metrics (structural role, off-hours work, and later affect/toxicity),
showing them in a UI.

Reference study design: **Option B** (structural role → strain → disengagement, Job Demands–Resources model).

## Technical choices

| Area | Choice | Rationale |
|---|---|---|
| Backend | Python 3.12, FastAPI, SQLAlchemy 2 | MSR/NLP ecosystem is Python |
| Database | SQLite (default, zero setup) or PostgreSQL (`DATABASE_URL`) | Same code; Postgres for large datasets |
| Job queue | Database table + separate `worker` process | No extra services (no Redis); queue and state live together |
| Commit mining | `git clone --bare` + `git log` | Fast, resumable by SHA |
| Issue/PR mining | GitHub REST API with `since` cursor | Incremental resume, rate-limit handling |
| Graphs | networkx | Centrality for core/peripheral |
| Frontend | React + Vite + TypeScript + Tailwind + Recharts | Modern UI, fast build |
| Deployment | `docker compose` (Postgres + api + worker + web) | Optional |

## Milestones

### M1 — Skeleton (this iteration)
1. Data model: projects, pipeline stages, logs, developers/aliases, commits, files, issues/PRs, comments, metrics.
2. Resumable runner: ordered stages, per-stage cursor, batch checkpoints, pause/resume/rerun,
   recovery of orphaned jobs after a crash.
3. Mining stages: clone, commits, GitHub commit authors, issues/PRs, comments (issue + review), reviews.
4. GitHub rate-limit handling (token pool, automatic wait until reset, status visible in the UI).
5. Processing stages: bot filter (heuristic), identity merge, structural metrics
   (truck factor, core/peripheral, off-hours), developer × month aggregation.
6. REST API + UI: Projects, Pipeline monitor, Project overview, Developers (pseudonymized).

### M2 — Affect
7. Sentiment (Senti4SD / SentiStrength-SE) and toxicity (ToxiCR) stages as plugins.
8. Emitted/received toxicity in the monthly aggregation.

### M1.5 — Export and correlation analysis (done)
- Pseudonymized ZIP export per repository, selection or all (CSV + codebook + metadata).
- Correlation analysis (Spearman/Pearson, partial, Benjamini–Hochberg) at developer or developer-month level.
- Threshold-based abandonment variables (`abandoned`, `exit`, `inactive_months_at_end`).

### M2.5 — Survival analysis (done)
- Counting-process dataset with delayed entry, lagged time-varying covariates and administrative censoring.
- Cox time-varying model (stratified, standardized, penalized) and Kaplan–Meier with log-rank test.
- `survival_intervals.csv` in the export.
- Cluster-robust SEs (person / subject / project) and Grambsch–Therneau PH test, validated against
  lifelines and numerical derivatives.

### M3 — Outcomes and analysis
9. Break states following Calefato et al. (beyond the simple abandonment threshold).
10. Mixed-effects models; time-varying effects (β(t)) for covariates that violate PH.
11. Interactive network view with time slider.

### M4 — Robustness
12. BoDeGHa for bots, more accurate identity merge, large-dataset tests, DB migrations (Alembic).

## M1 status
Implemented and verified end-to-end on `pallets/itsdangerous` and `pallets/markupsafe` (SQLite):
pause, worker stop/restart and resume continue from the saved cursor with no duplicated rows.
Docker setup is written but not yet tested.

## Definition of done for M1
- I add a URL from the UI, the pipeline starts, and I see per-stage progress.
- I pause, stop the worker, restart it, resume: no duplicated data, it restarts from the cursor.
- The overview shows activity, truck factor, core/peripheral and off-hours on a real repository.
