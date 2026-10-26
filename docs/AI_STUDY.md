# AI-generated contributions and open-source maintainers

Study design, measures and threats to validity. Every number reported here comes from the pipeline
and can be regenerated.

## 1. Goal

Measure whether, and how, the arrival of AI-generated contributions changes the work of open-source maintainers:
the load they carry, whether they stay, and whether newcomers who arrive with AI become lasting contributors.

Working title: *More Pull Requests, Fewer Maintainers? AI-Generated Contributions, Maintainer Workload and
Retention in Open Source.*

### The distinction the study is built on

| Source | Who triggers it | Expected effect on maintainers |
|---|---|---|
| **Agent PRs** (Copilot coding agent, Devin, Jules, Claude app) | People with write access assign the task: the agent is the *maintainer's tool* | Less effort per change, possibly more review |
| **Disclosed PRs** (a person's PR credits Claude Code, Codex, …) | Anyone, including first-time contributors | More review load if quality is low ("AI slop") |

The central question is whether AI **helps those who use it and weighs on those who receive it**.

## 2. Research questions

| RQ | Question | Measures | Method |
|---|---|---|---|
| RQ1 | How common are declared AI contributions in active projects, and how have they grown since 2023? | AI PRs and commits per month, by source and tool | Descriptive |
| RQ2 | After exposure starts, does maintainer load change? | Reviews and comments per maintainer, PR rejection rate, time to first response, off-hours share | Staggered event study (Callaway & Sant'Anna 2021), never/not-yet-exposed controls |
| RQ3 | Is exposure associated with maintainers leaving? Leaving the project vs leaving GitHub | Time-varying exposure in the Cox model; outcomes *abandonment* and *left GitHub* | Survival analysis (existing) |
| RQ4 | Do newcomers whose first PR is AI-assisted come back as often as others? | Second contribution within 6/12 months | Survival / logistic, matched on project and period |

## 3. Sample

The project list is provided by the authors. Each project is added with `since = 2023-01-01`
(`POST /api/projects {"urls": [...], "since": "2023-01-01"}`), which gives at least 18 months before the first
agent PRs (mid-2025) for the pre-trend check; the git history is always complete (local), so the truck factor uses
all of it.

Exposure is not taken from the list but measured on the mined data by the `ai_signals` stage, identically for every
project and month by month: a project's exposure starts in the first month with declared AI PRs, and projects
without any act as never-exposed controls (staggered event study).

Context from a first exploration (GitHub search, September–October 2026): agent PRs are tens of thousands per
month (Copilot coding agent: ~60,000 in September 2026), almost all in tiny personal repositories, while popular
projects with exactly no declared AI PR are rare (pallets/flask: 5 Claude Code PRs; home-assistant/core: 1,223).
Controls are therefore projects with *little* declared AI use, and a dose (share of AI PRs) is more informative
than a yes/no exposure.

## 4. Detecting AI involvement (`ai_signals` stage, `app/stages/ai.py`)

| Signal | Rule | Source |
|---|---|---|
| Agent account | Login in a fixed list (e.g. `copilot-swe-agent`, `devin-ai-integration`, `google-labs-jules`, `claude`); role *author* or *reviewer* (CodeRabbit, Copilot reviewer, Gemini Code Assist, …) | aliases |
| Commit credit | Co-author trailer with the tool's own address (`noreply@anthropic.com`, `…+Copilot@users.noreply.github.com`, `@cursor.com`, `@aider.chat`, …) or "Generated with Claude Code" | local git log |
| PR credit | Description links an agent session (`app.devin.ai/sessions`, `chatgpt.com/codex`, `cursor.com/agents`, `jules.google.com/task`) or says "Generated with Claude Code" | PR body |

Rules are deliberately narrow: names alone never count (a contributor called Jules, a commit that mentions
"copilot"), and tools that only write PR descriptions (Copilot summaries) are not code credits. Unit tests:
`backend/tests/test_ai.py`. **Precision must be checked by hand** on a random sample of 100 flagged PRs and
commits before reporting.

## 5. Threats to validity

- **Only declared AI is visible.** Undisclosed use is common, so exposure is a lower bound and controls are
  partly exposed: estimates are biased towards zero. A null result must be reported with this caveat.
- **Selection into exposure.** Projects that adopt agents differ from those that do not (company-backed, larger,
  faster). Matching and the pre-trend check reduce but do not remove this; results are associations, not causal
  effects, unless parallel pre-trends hold.
- **Exposure grows with activity.** More PRs mean more AI PRs: the analysis uses shares and project fixed effects,
  not raw counts.
- **No measure of wellbeing or code quality.** The study measures load and retention.
- **Generalisation:** active, popular GitHub projects created before 2023.
