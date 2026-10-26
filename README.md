# StrainScope

A research platform for studying, at the **individual developer** level, the relationship between
a developer's structural role in an open-source project, workload/strain signals, and disengagement.

Give it one or more GitHub URLs → it clones and mines each repository, reconstructs identities and
socio-technical networks, computes developer × month metrics, and shows them in a web UI.
Mining is **pausable and resumable**: the state of every stage is persisted in the database.

> Research tool. It does not measure burnout and must not be used to evaluate people. See [Ethics](#ethics-and-privacy).

---

## Contents
- [Theoretical grounding](#theoretical-grounding)
- [Metrics](#metrics)
- [Architecture](#architecture)
- [Pipeline](#pipeline)
- [Pause and resume](#pause-and-resume)
- [Data model](#data-model)
- [User interface](#user-interface)
- [Getting started](#getting-started)
- [Configuration](#configuration)
- [Analysis and export](#analysis-and-export)
- [API](#api)
- [Ethics and privacy](#ethics-and-privacy)
- [Known limitations and deviations from the papers](#known-limitations-and-deviations-from-the-papers)
- [Roadmap](#roadmap)

---

## Theoretical grounding

The design follows the **Job Demands–Resources** model (Bakker & Demerouti):
high demands → strain → disengagement.

| JD-R concept | Observable proxy in the repository |
|---|---|
| Demands | Being *core*, belonging to the *truck factor*, working off-hours |
| Strain | Negative affect, emitted/received toxicity *(milestone M2)* |
| Outcome | Abandonment (inactive ≥ N months at the end of observation); richer break states in M3 |

Unit of analysis: **developer × project × month**.

## Metrics

Each metric reuses a published operationalization. Check citations against the original papers
before using them in a publication.

| Metric | Operationalization | Reference |
|---|---|---|
| **Truck factor** | Degree of Authorship (DOA) per file; a developer is an author if normalized DOA > 0.75 and DOA ≥ 3.293; greedily remove the developers authoring the most files until > 50% of files are orphaned | Avelino et al., ICPC 2016 |
| **Core / peripheral** | Eigenvector centrality on the developer network (collaboration + communication), 3-month sliding window; core = developers covering 80% of cumulative centrality | Joblin et al., ICSE 2017 |
| **Off-hours** | Share of commits on weekends or outside working hours, in the author's own time zone | Claes et al., ICSE 2018 |
| Sentiment *(M2)* | Senti4SD / SentiStrength-SE | Calefato et al. 2018; Islam & Zibran 2018 |
| Toxicity *(M2)* | ToxiCR | Sarker et al., TOSEM 2023 |
| **Abandonment** | Inactive for ≥ `ABANDON_MONTHS` (default 12) before the project's last observed month; right-censored | Threshold approach; see Calefato et al., EMSE 2022 for break states *(M3)* |

DOA: `3.293 + 1.098·FA + 0.164·DL − 0.321·ln(1 + AC)`, where FA = the developer created the file,
DL = the developer's changes to it, AC = changes by others.

## Architecture

```
┌──────────── frontend/ (React + Vite + Tailwind + Recharts) ────────────┐
│  Projects · Pipeline monitor · Overview · Developers                    │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │ REST (polling every 2 s)
┌──────────────────────────────┴──────────────────────────────────────────┐
│  backend/  FastAPI (api)            backend/  worker (separate process) │
│  - creates projects                 - claims a queued project           │
│  - pause / resume / rerun           - runs stages in order              │
│  - serves metrics                   - saves cursor and checkpoints      │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                   SQLite (default) / PostgreSQL
           pipeline state · raw data · derived metrics
```

The API and the worker are separate processes: restarting the API does not interrupt mining, and
stopping the worker loses at most the last unsaved batch.

The queue is a database table — no Redis or message broker. Claiming a project is atomic
(`UPDATE ... WHERE status = 'queued'`), so several workers can run in parallel on PostgreSQL
without picking the same project.

## Pipeline

Every project goes through these stages, in order. Each stage is idempotent (upserts on natural keys):
re-running it never duplicates data.

| # | Stage | What it does | Resume cursor |
|---|---|---|---|
| 1 | `clone` | `git clone --bare` (or `fetch` if already present) | — |
| 2 | `commits` | Chronological `git log`: author, dates with time zone, touched files, lines, renames | last processed SHA |
| 3 | `commit_authors` | GitHub API: maps commit emails to GitHub logins | page |
| 4 | `issues` | Issues and PRs (`/issues?state=all&sort=updated`) | last `updated_at` |
| 5 | `comments` | Issue/PR comments and code-review comments | last `updated_at` per endpoint |
| 6 | `reviews` | PR reviews (approvals, change requests) via GraphQL, 100 PRs per request | GraphQL cursor |
| 7 | `bots` | Flags bot accounts (GitHub `Bot` type, `[bot]` suffix, known list) | — |
| 8 | `identities` | Merges aliases (email, name, login, GitHub noreply email) into developers | — |
| 9 | `aggregate` | Developer × month activity table (commits, lines, off-hours, issues, PRs, comments, reviews) | — |
| 10 | `structural` | Truck factor (at HEAD and month by month), developer network, centrality, core/peripheral (written into the monthly table) | — |

Stages 7–10 only compute on stored data: they are fast and can be re-run on their own
(e.g. after changing a threshold) without mining again.

## Completeness checks

GitHub's listing endpoints sometimes answer `200` with an empty or truncated page (reproducible with the
official `gh` CLI), which would silently drop data. The miner therefore:

- follows the `Link: rel="next"` URL (some endpoints use cursor pagination) and re-fetches suspicious
  pages with backoff: an empty page after a full one, a short page that still has a next link, or an
  empty first page (briefly, since empty first pages are often legitimate);
- **issues/PRs**: share one numbering 1..N, so every number missing from the listing is fetched
  individually; numbers that do not exist (deleted, transferred, discussions) answer 404/410 and are skipped;
- **issue comments**: GitHub reports each issue's comment count; threads with fewer saved comments are
  re-fetched one by one.

Review comments have no cheap count to check against and rely on the retries only.

**PR reviews via GraphQL.** Reviews are the most expensive data over REST (one request per PR). The
`reviews` stage fetches them with GraphQL, 100 pull requests per request, and falls back to REST for PRs
with more than 100 reviews, for PRs present in the issues listing but missing from GraphQL, and when no
token is configured (GraphQL requires one). GraphQL has its own quota, tracked separately.
Bot logins are mapped to the REST form (`name[bot]`). `backend/scripts/compare_reviews_rest_graphql.py`
compares both APIs field by field; results:

| Repository | PRs | Reviews | REST | GraphQL | Identical |
|---|---|---|---|---|---|
| pallets/itsdangerous | 312 | 47 | 312 requests, 148 s | 4 requests, 7 s | yes |
| pallets/markupsafe | 384 | 39 | 384 requests, 184 s | 4 requests, 7 s | yes |
| 10up/classifai | 766 | 1,392 | 766 requests, 434 s | 8 requests, 20 s | yes |
Recovered items are reported in the stage message and in the activity log.

## Text classification (sentiment and toxicity)

Two pipeline stages classify every issue/PR (title + body), comment and review written by a human. Text is
cleaned first so that only the author's own words are scored: quoted replies (`> ...`), fenced code, HTML
comments (issue templates) and @mentions are removed. Bots and empty texts are skipped. Results feed
monthly *emitted* and *received* measures (received = messages by others on the developer's own issues/PRs),
which are available as correlation variables and Cox covariates. Unmeasured values stay missing, never zero.

The tools run locally in their own environments under `external/` (ignored by git; their licenses stay
separate), as persistent subprocesses (`backend/app/tools/*_runner.py`).

| Stage | Tool | Status |
|---|---|---|
| `sentiment` | Senti4SD (Calefato et al., EMSE 2018), learner retrained on the GitHub gold standard of Novielli et al. (MSR 2020) | **Active**; 10-fold macro-F1 **0.89** on GitHub |
| `toxicity` | ToxiCR (Sarker et al., TOSEM 2023), random forest on TF-IDF with keyword removal | **Off by default** (`ENABLE_TOXICITY=true`): F1 0.37 on GitHub |

Stages skipped because a tool was missing run again automatically once it is installed (worker start-up).

**Senti4SD validation.**

| Check | Result |
|---|---|
| Published model, Stack Overflow test set, `Senti4SD.jar` features, numpy prediction | accuracy **0.8748**, identical to the published 0.8748 (prediction reimplementation is exact) |
| Same, `Senti4SD-fast.jar` features (the pipeline's default, used here for speed) | accuracy 0.827 |
| Published (Stack Overflow) model on the GitHub gold standard (7,122 comments) | macro-F1 0.70 |
| **Learner retrained on the GitHub gold standard, 10-fold CV** | **macro-F1 0.892** (negative 0.885, neutral 0.890, positive 0.901) |
| GitHub-retrained model on the Stack Overflow test set (out of domain) | accuracy 0.755 |

Retraining within the platform is what Novielli et al. recommend when a gold standard exists. The Java
extractor must run with an English locale and UTF-8 (`-Duser.language=en -Dfile.encoding=UTF-8`): with the
system's Italian locale accuracy dropped by 0.6 points. Prediction uses `backend/app/tools/senti4sd_model.py`,
a numpy implementation of LIBLINEAR's linear prediction, so the Linux-only `liblinear.so` shipped with
pySenti4SD is not needed.

**ToxiCR validation.** Its 10-fold cross-validation was reproduced on this machine with the authors' code
(F1 on toxic comments 0.865 vs 0.872 published, on Gerrit code reviews). On GitHub it does not transfer:

| Evaluation on GitHub issue comments (STRUDEL dataset, Raman et al. 2020: 1,597 comments, 101 toxic) | P | R | F1 |
|---|---|---|---|
| ToxiCR trained on code reviews | 0.54 | 0.28 | 0.37 |
| ToxiCR retrained on GitHub only (10-fold CV) | – | 0.00 | 0.00 |
| ToxiCR retrained on code reviews + GitHub (10-fold CV) | 0.54 | 0.27 | 0.36 |

The STRUDEL dataset is 6% toxic; ordinary repositories are around 0.6% (Miller et al., ICSE 2022). With a
false-positive rate of about 1.6% and recall 0.28, roughly nine in ten flagged messages would be false
positives at that base rate, which matches a manual check on 10up/classifai (13 of 13 false positives).
Per-developer toxicity would be mostly noise, so the stage stays disabled. A usable measure needs a larger
GitHub gold standard (e.g. the locked-issue incivility dataset of Ehsani et al., MSR 2024) or manual annotation.

Installing Senti4SD (pySenti4SD's DagsHub storage requires an account; the identical files are on the
original repository's Git LFS):

```bash
cd external
git clone https://github.com/collab-uniba/pySenti4SD.git
(cd pySenti4SD/java && python3 ../../../backend/scripts/fetch_senti4sd_lfs.py)   # ~960 MB, SHA-256 + MD5 checked
python3.11 -m venv senti4sd-venv && senti4sd-venv/bin/pip install "numpy<2" pandas scipy scikit-learn joblib
mkdir -p gold && curl -L -o gold/github_gold.csv https://ndownloader.figshare.com/files/21001260   # MD5 3166bb1f…
mkdir -p senti4sd-work/features
python3 -c "import csv; r=list(csv.reader(open('gold/github_gold.csv',encoding='utf-8'),delimiter=';')); w=csv.writer(open('senti4sd-work/gh_gold.csv','w',newline='',encoding='utf-8'),delimiter=';',quoting=csv.QUOTE_ALL); w.writerow(['ID','Polarity','Text']); [w.writerow([x[0],x[1],x[2].rstrip('\"').replace('\n',' ')]) for x in r[1:]]"
(cd pySenti4SD && ../senti4sd-venv/bin/python python/csv_processing.py -i ../senti4sd-work/gh_gold.csv -d sc -c text)
java -Duser.language=en -Duser.country=US -Dfile.encoding=UTF-8 -jar pySenti4SD/java/Senti4SD-fast.jar -F A \
  -i senti4sd-work/gh_gold_jar.csv -W pySenti4SD/java/dsm.bin -oc senti4sd-work/features/gh_gold.csv -vd 600
senti4sd-venv/bin/python ../backend/scripts/train_senti4sd_github.py
```

Installing ToxiCR (BERT needs TensorFlow 2.5, unavailable on Apple Silicon; the random forest needs only
scikit-learn):

```bash
cd external
git clone https://github.com/WSU-SEAL/ToxiCR.git
python3.10 -m venv toxicr-venv
toxicr-venv/bin/pip install numpy==1.21.6 pandas==1.3.5 scikit-learn==1.0.2 sklearn-pandas==2.2.0 nltk==3.8.1 openpyxl==3.1.2
toxicr-venv/bin/python -m nltk.downloader -d nltk_data punkt
cd ToxiCR && NLTK_DATA=../nltk_data ../toxicr-venv/bin/python ToxiCR.py --algo RF --embed tfidf --keyword --mode pretrain
```

New stages are added to existing projects automatically when the worker starts: mining stages are kept and
only the new stages and those after them (aggregation, structural metrics) are recomputed.

## Pause and resume

- **Pause**: the UI sets `pause_requested`. The worker checks the flag after every batch, saves the
  cursor, and marks the project `paused`.
- **Resume**: the project goes back to `queued`; the worker restarts from the first unfinished stage, at its cursor.
- **Rerun a stage**: the chosen stage and every later stage go back to `pending`.
- **Delete**: queued, paused, failed and finished projects are deleted at once (rows and clone). A project
  that is being mined becomes `deleting`: the worker stops at its next checkpoint and deletes it itself, so
  the API never removes data or clones under a running stage. If the worker is not alive, the project is
  deleted at once; pending deletions are also completed when a worker starts.
- **Crash or stopped worker**: the worker writes a heartbeat. At startup, `running` projects with a
  stale heartbeat go back to `queued` and resume from their last checkpoint.
- **GitHub rate limit**: when a token runs out the next one is used; if all are exhausted the worker
  waits for the reset (shown in the UI) and then continues on its own. Pausing still works while waiting.
- **Stopping the worker** (Ctrl+C / SIGTERM): the current batch is discarded, the project goes back to
  `queued`, and the next worker run continues from the last checkpoint. Press Ctrl+C twice to force-quit.

## Data model

| Table | Contents |
|---|---|
| `projects` | Repository, status, pause flag, heartbeat |
| `stage_runs` | Status, progress and cursor (JSON) of every stage of every project |
| `logs` | Pipeline logs shown in the UI |
| `developers` | Unified developer, pseudonym, bot flag |
| `aliases` | (name, email) pairs and GitHub logins linked to a developer |
| `commits`, `commit_files` | Commits and touched files (change type, lines, renames) |
| `issues` | Issues and PRs |
| `comments` | Issue/PR comments, review comments, review bodies |
| `developer_months` | Aggregated developer × month metrics |
| `project_metrics` | Truck factor and other project-level metrics |
| `app_state` | Global state (worker heartbeat, pseudonym salt) |

The GitHub quota shown in the UI is written to `DATA_DIR/github_rate.json` rather than the database:
with SQLite a second writer would block on the stage's open transaction.

## User interface

1. **Projects**: paste one or more URLs (one per line) and see the status of every repository. Each card
   can be paused, resumed or deleted; **Delete all** removes every project after typing `delete` to confirm.
2. **Pipeline monitor**: per-stage progress, pause / resume / rerun, live logs, GitHub quota.
3. **Overview**: monthly activity, core developers over time, truck factor, commit-hour heatmap.
4. **Developers**: pseudonymized table with role, activity, off-hours share, and monthly timeline.
5. **Correlations**: correlation matrix across one, several or all projects (see below).
6. **Survival**: Cox model on time-varying covariates and Kaplan–Meier curves for time to abandonment.
7. **Export**: download a dataset for one repository, a selection, or everything.
8. **Settings**: add, check and remove GitHub tokens.

## Analysis and export

### Correlation analysis
- **Unit of analysis**: *developer* (one row per developer × project, aggregated) or *developer-month* (panel).
- **Methods**: Spearman (default, robust for skewed counts) or Pearson. With a binary variable they reduce
  to rank-biserial / point-biserial correlations.
- **Partial correlation**: pick control variables (e.g. commits, tenure) to remove their linear effect.
  Both variables are ranked for Spearman, regressed on the (ranked) controls, and the residuals are correlated;
  the t-test uses n − 2 − k degrees of freedom.
- **Multiple testing**: Benjamini–Hochberg q-values over all pairs in the matrix; `*` marks q < 0.05.
- **Missing values**: pairwise deletion (e.g. off-hours share is undefined for developers without commits),
  listwise on the variables involved when controls are used. `n` is reported per pair.
- **Filters**: projects, minimum active months, abandonment threshold.
- **Warnings** are shown when p-values are not trustworthy: developer-month rows are repeated measures
  (not independent), and pooling several projects mixes within- and between-project variation.
  For inference on the panel, export the data and fit a mixed-effects or survival model.
- Click a cell for the scatter plot of the pair; the pair table can be downloaded as CSV.

Results were checked against `scipy.stats` (Spearman, Pearson, `false_discovery_control`) and the analytic
first-order partial-correlation formula.

### Abandonment variables
| Level | Variable | Meaning |
|---|---|---|
| developer | `inactive_months_at_end` | Months between last activity and the project's last observed month |
| developer | `abandoned` | 1 if `inactive_months_at_end ≥ ABANDON_MONTHS` |
| developer | `long_pauses` | Pauses of ≥ `ABANDON_MONTHS` after which the developer came back |
| developer-month | `exit` | 1 in the last active month of a developer who then abandoned |
| developer-month | `pause_start` | 1 in an active month followed by a pause of ≥ `ABANDON_MONTHS` and a return |

Right-censoring: developers who left less than the threshold before the end of observation count as not
abandoned. One-off contributors trivially "abandon"; use the minimum-active-months filter to exclude them.

### Survival analysis (time to abandonment)
Counting-process design, one subject per developer × project, time in calendar months since entry:

- **Entry** after the developer's N-th active month (left truncation; default N = 1). Filtering on *total*
  active months would use future information (immortal-time bias), so the filter is implemented as delayed entry.
- **Risk set = active months only.** Each row asks "after this month, does the developer stop?". During a
  pause the developer is not at risk and re-enters when they come back. Keeping pause months at risk would
  teach the model that inactivity protects from leaving, because those months can never be event months.
- **Event**, selectable in the UI:
  - **Abandonment** (default): the developer stops and never comes back. A pause followed by a return is
    never abandonment, however long. The threshold K (`ABANDON_MONTHS`, default 12) is needed only because
    the future is unknown: a final silence counts as abandonment only if it lasts ≥ K months before the end
    of the data; more recent exits are censored (they may still return).
  - **Any long pause**: every pause of ≥ K months is an event, with or without return. A developer can have
    several (recurrent events, Andersen–Gill), so standard errors should be clustered by person.

  Example: active until January 2020, silent for all of 2020, back in February 2021 and still active.
  Abandonment: no event, never at risk during 2020. Long pause: one event in January 2020, at risk again
  from February 2021.
- Pauses that start in the entry month are not at risk (the entry month never is), so the survival counts
  can be lower than the `long_pauses` variable of the correlation dataset.
- **Time-varying covariates** for month t summarize months t−W…t−1 (default W = 3), lagged so the activity
  that marks the exit month never enters its own covariates: core (any month), max centrality, off-hours
  share, commits, comments, reviews, issues/PRs opened, active months in the window, cumulative experience.
- **Truck-factor membership (time-varying)**, on by default: the `structural` stage replays the commit history
  and stores the truck factor at the end of every month (`truck_factor_history`), computed from the commits made
  up to then and the files alive at that point; month t uses the snapshot of t−1. Its last snapshot equals the
  HEAD-based truck factor on all 100 mined projects. The older fixed membership (measured at the last commit,
  look-ahead bias) is still available, off by default.
- **Project decline** (controls, on by default): distinct *other* developers active in the project in the window,
  and the activity trend log((others' activity + 1) / (others' activity in the previous window + 1)). Without
  them, developers who leave a dying project are indistinguishable from individual disengagement.
- **Responses received** (off by default), on the developer's **issues** opened in the window: share with no comment
  or review by another human within 30 days (0 when none opened, so include *Issues + PRs opened*), and median
  days to the first response (unanswered = 30; intervals with no issue opened are excluded). Pull requests are
  left out: merges are not mined, and 46–61% of PRs get no comment, often because they are merged silently.
- **Model**: `lifelines.CoxTimeVaryingFitter`, optionally stratified by project (separate baseline hazards),
  with log1p for counts, standardization (HR per SD) and an optional L2 penalizer.
- **Cluster-robust standard errors** (Lin & Wei sandwich, as R's `coxph(..., cluster = id)`), clustered by
  person (GitHub login, so the same person in several repositories is one cluster), developer × project,
  or project; model-based SEs are shown alongside. CIs and p-values use the robust SEs. A warning appears
  with fewer than 30 clusters.
- **Proportional-hazards test** (Grambsch & Therneau 1994): per covariate and global χ² on Schoenfeld residuals,
  with g(t) = rank (default), log or identity — the classic statistic of `cox.zph` before survival 3.0 and of
  lifelines' `proportional_hazard_test`. Covariates with p < 0.05 are flagged.
- **Diagnostics in the UI**: events per covariate < 10, (quasi-)separation, few clusters, PH violations,
  unstratified pooling.
- **Kaplan–Meier** curves of time to the *first* event, by a grouping measured at entry (core, off-hours above/below median,
  truck-factor membership at entry, or the fixed truck factor with its look-ahead caveat) with a log-rank test.
Estimates come from `lifelines.CoxTimeVaryingFitter` (Efron ties). Score and Schoenfeld residuals,
the information matrix, the sandwich variance and the PH test are computed in `backend/app/coxdiag.py`
with Efron's tie correction, following R's `survival` definitions. lifelines' own score residuals ignore
ties and `statsmodels.PHReg` counts rows entering exactly at an event time as at risk, so neither is used
for these quantities.

Validation:
- `backend/tests/test_survival.py` checks the risk set and event definitions on hand-made timelines
  (pause with return, short pause, censoring near the end, one-off contributors, delayed entry, lagged
  covariates); run with `python -m pytest backend/tests` after `pip install -r requirements-dev.txt`;
- the time-varying fitter matches a direct maximization of the Efron partial likelihood, and with a
  time-fixed covariate it reproduces `CoxPHFitter` exactly (coefficient and SE);
- model-based SEs from `coxdiag` equal lifelines' to 6 decimals;
- with ties (50 of 76 events), the summed score residuals equal the numerical gradient of the Efron
  log partial likelihood and the information equals the numerical Hessian (max abs. difference 4e-4);
- without ties, robust SEs equal `CoxPHFitter(robust=True, cluster_col=...)` and PH statistics equal
  `proportional_hazard_test` for the rank, log and identity transforms, to 5–6 decimals;
- every subject has at most one event, on its last interval.

Not included: the exact score test of `cox.zph` in survival ≥ 3.0, time-interaction terms, and
small-sample corrections for the sandwich. The exported `survival_intervals.csv` supports these in R.

### Export
A ZIP with `projects.csv`, `developers.csv`, `developer_months.csv`, `survival_intervals.csv`
(counting-process table with default settings), `codebook.md` (every column defined) and `metadata.json`
(parameters, row counts). Exports never contain names, logins, emails or message text,
even when `SHOW_REAL_IDENTITIES` is on.

## Getting started

### Local (no Docker)

Requirements: Python ≥ 3.11, Node ≥ 20, git.

```bash
# backend
cd backend
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export GITHUB_TOKENS=ghp_xxx                  # one or more tokens, comma-separated
uvicorn app.api:app --reload --port 8000      # terminal 1
python -m app.worker                          # terminal 2

# frontend
cd frontend
npm install
npm run dev                                   # http://localhost:5173
```

**GitHub tokens**: add them from the UI (**Settings → Add a GitHub token**). Each token is checked with
GitHub before being saved, and the worker picks up added or removed tokens within 30 seconds, without a
restart. A fine-grained token with read-only access to public repositories is enough.
Alternatively set `GITHUB_TOKENS` (environment or `backend/.env`); if the GitHub CLI is logged in you can use
`export GITHUB_TOKENS=$(gh auth token)`. Environment and UI tokens are combined.

Single-port mode: after `npm run build` in `frontend/`, the API also serves the UI at http://localhost:8000.

### Docker

```bash
GITHUB_TOKENS=ghp_xxx docker compose up --build    # http://localhost:8080
```

Starts PostgreSQL, the API (which also serves the built UI), and the worker.
Use SQLite with a single worker; with PostgreSQL you can scale workers (`docker compose up --scale worker=3`).

## Configuration

Environment variables (a `.env` file in `backend/` is also read):

| Variable | Default | Meaning |
|---|---|---|
| `DATABASE_URL` | `sqlite:///./data/strainscope.db` | Database (e.g. `postgresql+psycopg://user:pw@host/db`) |
| `DATA_DIR` | `./data` | Repository clones |
| `GITHUB_TOKENS` | — | GitHub tokens, comma-separated (combined with tokens added in Settings) |
| `BATCH_SIZE` | `200` | Items per checkpoint |
| `WORK_HOURS` | `9-18` | Working-hours window for the off-hours metric |
| `CORE_THRESHOLD` | `0.8` | Share of cumulative centrality defining core developers |
| `NETWORK_WINDOW_MONTHS` | `3` | Network window width |
| `ABANDON_MONTHS` | `12` | Inactivity threshold for abandonment (overridable per analysis in the UI) |
| `PSEUDONYM_SALT` | random, stored in `app_state` | Salt for pseudonyms |
| `SHOW_REAL_IDENTITIES` | `false` | Show real names in the UI (local use only) |

## API

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/projects` | `{ "urls": ["https://github.com/o/r", ...] }` |
| `GET` | `/api/projects` | List with status and progress |
| `GET` | `/api/projects/{id}` | Detail with stages |
| `POST` | `/api/projects/{id}/pause` | Request a pause |
| `POST` | `/api/projects/{id}/resume` | Resume |
| `POST` | `/api/projects/{id}/rerun/{stage}` | Re-run from a stage |
| `DELETE` | `/api/projects/{id}` | Delete a project and its data (deferred while it is being mined) |
| `DELETE` | `/api/projects` | Delete all projects |
| `GET` | `/api/projects/{id}/logs` | Recent logs |
| `GET` | `/api/projects/{id}/overview` | Project-level metrics |
| `GET` | `/api/projects/{id}/developers` | Per-developer metrics |
| `GET` | `/api/projects/{id}/developers/{dev}` | Monthly timeline of one developer |
| `GET` | `/api/export?project_ids=1,2` | Dataset ZIP (omit `project_ids` for all projects) |
| `GET` | `/api/analysis/variables` | Variables per unit of analysis, defaults |
| `POST` | `/api/analysis/correlations` | Correlation matrix, pairs, warnings |
| `POST` | `/api/analysis/scatter` | Points for one pair (sampled above 3000) |
| `GET` | `/api/survival/options` | Covariates, groupings, defaults |
| `POST` | `/api/survival` | Risk-set summary, Cox model, Kaplan–Meier curves |
| `GET` | `/api/settings/tokens` | Configured tokens (last 4 characters only) |
| `POST` | `/api/settings/tokens` | `{ "token": "..." }`: validate with GitHub and save |
| `POST` | `/api/settings/tokens/{id}/check` | Re-validate a saved token |
| `DELETE` | `/api/settings/tokens/{id}` | Remove a saved token |
| `GET` | `/api/status` | Worker status and GitHub quota |

## Ethics and privacy

Inferring wellbeing-related states about identifiable people is sensitive even on public data
(GDPR, notably Art. 9, and Art. 89 for research).

- **Pseudonyms by default**: salted hashes. Real names appear only with `SHOW_REAL_IDENTITIES=true`.
- **Everything local**: no data leaves the machine. The API has no authentication, so it only accepts
  cross-origin requests from the dev UI (CORS), and Docker publishes it on 127.0.0.1 only.
- **Tokens** added in Settings are stored in plain text in the local database (like a `.env` file) and are
  never returned by the API: only their last four characters are shown.
- **No sharing features** for individual profiles.
- **Export** is pseudonymized only, with no raw text.
- Before a study: a DPIA, and ethics-board approval if repository data is combined with surveys.

The data is **pseudonymized, not anonymous**: longitudinal per-developer data is re-identifiable,
so the GDPR still applies.

## Known limitations and deviations from the papers

To be reported among threats to validity.

- **File-level collaboration network**: Joblin et al. use function-level co-edits.
  Here two developers are linked if they change the same file in the same window.
- **Communication network**: developers are linked if they take part in the same issue/PR thread.
- **Truck factor**: computed on the files present at the last commit, without the third-party-file
  filter (Linguist) used by Avelino et al.
- **Bots**: heuristic, not BoDeGHa (roadmap).
- **Identities**: simplified heuristics compared with Bird et al.; merges may be missed or wrong.
- **Off-hours**: relies on the commit's declared time zone, which may be the server's.
- **Partial visibility**: chat, mailing lists and private work are not observed.
- **Small windows**: in months with very few active developers almost everyone is classified as core.
  Consider a minimum network size before using the core flag in a model.

## Roadmap

See [docs/PLAN.md](docs/PLAN.md). In short:
- **M2**: sentiment and toxicity as plugin stages.
- **M3**: break states (Calefato et al.), mixed-effects models, time-varying effects for covariates that violate PH.
- **M4**: BoDeGHa, more accurate identities, DB migrations.
