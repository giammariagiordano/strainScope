# Toxicity measurement in StrainScope

This document records how toxicity is measured, why, and how the classifier was built and validated. Every number
below comes from a script in `backend/scripts/toxicity/` and can be regenerated (see [Reproduction](#reproduction)).

## 1. Decision history

| Date | Step | Outcome |
|---|---|---|
| 2026-10-22 | ToxiCR (Sarker et al., TOSEM 2023), random forest, trained on Gerrit code reviews | Published 10-fold CV reproduced (F1 0.865 vs 0.872). On GitHub: 13 of 13 flagged comments in 10up/classifai were false positives. Stage disabled. |
| 2026-10-23 | ToxiCR on GitHub issue comments (STRUDEL dataset, Raman et al. 2020: 1,597 comments, 101 toxic) | P 0.54, R 0.28, F1 0.37; retraining on 101 toxic examples fails (F1 0.00). |
| 2026-10-23 | Literature search for GitHub-validated detectors | Most evaluations use "heated" threads (6–28% toxic). Only the ASE 2026 study evaluates at the natural rate (0.76%). |
| 2026-10-23 | Unit of measurement changed from **developer** to **conversation / community** | Matches the available labels (conversation level), avoids labelling individuals as toxic, and corresponds to a community climate (cf. the *Black Cloud* smell). |
| 2026-10-23 | Train a BERT-family classifier on the ASE 2026 data, validate at the natural rate | Sections 3–6. |
| 2026-10-26 | StrainScope has only 5,985 conversations with human text (88 repositories): the ASE pipeline itself (LLM + verifier + human review) is affordable on them | Section 7. The fine-tuned classifier is kept as a baseline and for larger samples. |
| 2026-10-26 | **Toxicity work paused.** Measured cost: Ministral-3-14B on the 16 GB M1 is estimated at 15–30 s per conversation (≈ 25–50 h for StrainScope, 4–7 h for the validation), plus human review; with ~45 toxic conversations expected the covariates would have little statistical power. | Nothing run beyond the scripts' dry run. Strain is measured by sentiment (Senti4SD). The pipeline runs without toxicity: ToxiCR is off, `conversation_toxicity` skips itself without a model, the toxicity covariates are off by default. To resume: section 7, starting with the validation on the 898 human-labelled conversations. |

### Detectors considered

| Detector | Evaluated on | Result on GitHub | Why not used as is |
|---|---|---|---|
| ToxiCR (Sarker et al., TOSEM 2023) | Gerrit code reviews | F1 0.37 (STRUDEL) | Does not transfer to GitHub |
| STRUDEL (Raman et al., ICSE NIER 2020) | GitHub issue comments | F1 ≈ 0.40 (authors' CV) | Weak |
| BERT incivility (Ferreira et al., JSS) | Locked GitHub issues, LKML | F1 > 0.9 within dataset, poor cross-platform | Incivility ≠ toxicity; heated threads only |
| Small LLMs (SBES 2025) | 6,974 comments from heated GitHub threads | best local F1 ≈ 0.63 (Gemma 2 9B) | Heated threads only |
| ChatGPT zero-shot (Mishra & Chatterjee, ICSE NIER 2024) | STRUDEL | "promising" | Text leaves the machine; model versions change |
| LLM + verifier + human review (ASE 2026, Zenodo 10.5281/zenodo.21231251) | 124,757 GitHub conversations, natural rate | 99.9% estimated accuracy after review | 14B LLM: 30–80 h on this machine; needs human review |

## 2. Data

**Source.** Replication package of an ASE 2026 paper (double-blind, authors listed as "Anonymous"):
Zenodo record [10.5281/zenodo.21231251](https://doi.org/10.5281/zenodo.21231251), file `backup.zip`,
309,999,633 bytes, MD5 `8cfc7e3aa69c60d071c7dfe2f35a97a3` (verified after download), licence CC BY 4.0
(code inside: MIT). Stored in `external/toxicity-gh/` (ignored by git).

**Labels.**

| File | Unit | Size | Toxic | How labelled |
|---|---|---|---|---|
| `large-dataset/toxicity_label_complete_124k.csv` | conversation | 124,757 (340 repositories) | 946 (0.76%) | Ministral-3-14B labels, a random-forest verifier flags likely errors, humans review the flagged 1.5% (κ = 0.78 on 200). Authors estimate 99.90% accuracy. |
| `derailment-paper-data/unified_final_dataset_conversations.csv` | conversation | 898 | 202 | Fully human-labelled (Imran et al.) |
| `derailment-paper-data/unified_final_dataset_comments.csv` | comment | 8,873 | 446 | Fully human-labelled |

**Important caveat.** The 124k labels are not all human-checked: the LLM labelled everything and humans reviewed
only what the verifier flagged. A model trained on them partly learns the LLM's judgement. The LLM changed 169
labels after review: it had 14 false positives and 155 false negatives against the final labels. For this reason
the model is also evaluated on the 898 fully human-labelled conversations.

**Definition used by the labels** (prompt in the package): *"A conversation is toxic if it contains rude,
disrespectful, dismissive, antagonistic, profane, mocking, or hostile language that could make a reasonable GitHub
contributor feel unwelcome … Harsh technical criticism can be non-toxic if it stays professional."*

## 3. Preparation (`prepare_data.py`)

- Conversation text as provided ("user: comment" per line, issue/PR body first).
- **Speaker names → `P1:`, `P2:` …** (only the names known from the package's comment table, so URLs and lines
  like "Note:" are untouched) and **@mentions → `@user`**: the model must learn from what is said, not who says it,
  and StrainScope removes mentions before classifying.
- **Split by repository** (70/15/15, `GroupShuffleSplit`, seed 13): test repositories are never seen in training
  or threshold selection, as in real use on new projects.

| Split | Conversations | Repositories | Toxic | Rate |
|---|---|---|---|---|
| train | 93,269 | 238 | 725 | 0.78% |
| validation | 16,796 | 51 | 117 | 0.70% |
| test | 14,692 | 51 | 104 | 0.71% |

No repository overlaps with the 100 projects mined by StrainScope. Conversation length: median 1,880 characters,
90th percentile 7,664, maximum 391,854.

## 4. Evaluation protocol (`evaluate.py`)

- **Metrics on the toxic class:** precision, recall and F1 at a threshold chosen on the validation split (maximum F1
  at the natural rate), applied unchanged to test. Plus **average precision (PR-AUC)**, the threshold-free summary
  that matters when positives are below 1%.
- **95% CIs** by bootstrap over test repositories (1,000 resamples), because repositories are the sampling unit.
- **External check:** the same threshold on the 898 human-labelled conversations (toxic rate 22%, a "heated" sample,
  so precision there is not comparable with the natural rate).

## 5. Results

### Baselines

| Model | Test P | Test R | Test F1 | Test AP | Imran F1 |
|---|---|---|---|---|---|
| LLM alone (Ministral-3-14B, before review)¹ | 0.967 [0.92–1.00] | 0.856 [0.77–0.92] | 0.908 [0.85–0.95] | 0.829 | — |
| TF-IDF (word 1–2 + char 3–5 grams) + logistic regression | 0.182 [0.13–0.24] | 0.192 [0.11–0.27] | 0.187 [0.13–0.23] | 0.119 | 0.719 |

¹ Optimistic: the reference labels were derived from these predictions. It is the upper reference, not a fair
competitor.

TF-IDF illustrates the base-rate problem: reasonable on the heated sample (F1 0.72), poor at the natural rate (0.19).

## 6. Classifier (`train_transformer.py`, `windows.py`)

| Choice | Value | Why |
|---|---|---|
| Initialisation | `unitary/unbiased-toxic-roberta` (Detoxify, RoBERTa-base trained on Jigsaw, Apache-2.0) | Fewer than 1,000 toxic training examples: start from a model that already encodes toxicity. Its zero-shot use is also a baseline. |
| Head | single logit, initialised from Detoxify's `toxicity` output | Training starts from the zero-shot behaviour and adapts it to GitHub. |
| Long conversations | windows of 512 tokens (64 overlap), at most 4 (≈ 2,000 tokens, ≈ 90% of conversations fully covered); score = max over windows | A conversation is toxic if any part of it is. Text beyond ~12,000 characters is not read (threat to validity, see below). |
| Loss | binary cross-entropy on the max-over-windows logit (multiple-instance learning) | Labels are per conversation, not per window. |
| Sampling | each epoch: all 725 toxic training conversations + 10× as many non-toxic, re-sampled per epoch | Keeps epochs affordable on a laptop; the threshold is chosen afterwards at the natural rate. |
| Optimisation | AdamW, lr 2e-5, weight decay 0.01, 6% warm-up, linear decay, 3 epochs, 2 conversations × 8 accumulation steps, gradient clipping 1.0, seed 13 | Standard fine-tuning settings for RoBERTa-base. |
| Model selection | validation average precision at the natural rate, after each epoch | The test split is used once, at the end. |
| Hardware | Apple M1, 16 GB, PyTorch 2.5.1 MPS backend; inference batches of 8 windows (32 ran out of GPU memory) | — |

<!-- results of zero-shot Detoxify and of the fine-tuned model are added below by the next steps -->

## 7. Labelling StrainScope's conversations with the ASE pipeline (paused, not run)

With ~6,000 conversations the classifier is not needed to make labelling affordable: the pipeline that produced the
reference labels can be applied directly, so StrainScope's labels have the same definition and the same quality
control as the data, and are human-checked where the verifier doubts the LLM.

| Step | Script | What it does |
|---|---|---|
| Export | `export_conversations.py` (backend venv) | Conversations built exactly as the `conversation_toxicity` stage builds them (`conversation_text`): body + conversation comments, human authors, `P1:`… speakers, `@user` mentions. 5,985 conversations from 88 repositories; median 890 characters, p99 15,498. |
| LLM | `llm_label.py` (toxbert-venv, Ollama) | Ministral-3-14B (`ministral-3:14b`, Q4_K_M, 9.1 GB) with the package's category prompt verbatim: toxicity label, explanation, eight event-category scores. Resumable JSON lines. |
| Verifier | `verify_labels.py` (toxbert-venv) | The package's random forest and scaler (scikit-learn 1.5.2) estimate whether each LLM label is correct; p < 0.5 → `<name>_review.csv` for human review. |
| Review | by hand | Fill `human_label` for the flagged conversations; the final label is the human one where present, the LLM's otherwise. |

**Deviations from the package**, which does not record its inference settings: temperature 0 and seed 13
(reproducibility); JSON-schema constrained decoding (every reply parses); conversations cut at 20,000 characters so
prompt + conversation fit in an 8,192-token context on a 16 GB machine (> 99% of conversations are not cut; cut ones
are flagged).

**Validation of the local setup.** Before labelling StrainScope, the same scripts run on the 898 human-labelled
conversations, which the package also labelled with Ministral: agreement with the package's predictions shows
whether quantization and settings reproduce its LLM, and agreement with the humans gives the quality to report.
Reference, from the package's own outputs on those 898 conversations: LLM alone P 0.98, R 0.62, F1 0.76
(κ 0.71); after reviewing the 77 conversations (8.6%) the verifier flags, F1 0.98. The latter is optimistic: the
verifier was trained on these same conversations.

**Statistical power.** At the natural rate (~0.75%) about 45 of the 5,985 conversations are expected to be toxic.
Per-developer exposure (`toxic_threads`) will be zero for almost every developer-month; `community_toxicity` is the
more usable variable, and toxicity results should be presented as exploratory unless the sample grows.
