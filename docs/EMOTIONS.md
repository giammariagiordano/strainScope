# Emotion detection in StrainScope

How the `emotions` stage labels messages with Shaver's six basic emotions (anger, joy, love, sadness, fear,
surprise), why RoBERTa was chosen, and how it was validated. Every number comes from a script in
`backend/scripts/emotions/` and can be regenerated.

## 1. Decision history

| Date | Step | Outcome |
|---|---|---|
| 2026-10-26 | EmoTxt / EMTk (Calefato et al., ACII 2017) chosen: SE-specific, same group as Senti4SD | Its default models need politeness and mood features from Python 2 code with 2017 pickles; retrain without them. |
| 2026-10-26 | Retrain on EmoTxt's Stack Overflow gold standard (Novielli et al., MSR 2018; 4,800 posts) | The Java feature extraction recomputes each term's IDF by re-reading every document: hours. Features reimplemented in Python and **verified identical** (§2). |
| 2026-10-26 | Validation on GitHub through polarity (Novielli et al., MSR 2020; 7,122 comments) | SO-trained EmoTxt: macro-F1 0.59, κ 0.39 (precision 0.77–0.79, recall 0.37–0.47). Too low. |
| 2026-10-26 | GitHub emotion gold standard found (Imran et al., ASE 2022; 2,000 comments, κ > 0.8) | EmoTxt trained on SO + GitHub: micro-F1 0.49 on the official GitHub test, vs 0.42 for the original EMTk there. |
| 2026-10-26 | RoBERTa-base fine-tuned on the same data | Micro-F1 0.60 on the same test, better on all six emotions; polarity macro-F1 0.75, κ 0.62. Thresholds from only 320 comments over-report anger on natural data. |
| 2026-10-26 | Augmented data of Imran et al., thresholds from 5-fold out-of-fold predictions | 5-fold ensemble 0.63, single fold models 0.61–0.65. **Final model**: one RoBERTa on all training data with those thresholds, micro-F1 **0.62**, polarity 0.76 (κ 0.63), 5× faster than the ensemble. **Used by the stage.** |

## 2. EmoTxt features, reimplemented and verified (`app/tools/emotxt_features.py`)

EmoTxt's features are tf·idf of uni- and bigrams of a corpus dictionary and of WordNet Affect categories (each token
replaced by its category). The Python code follows the Java source line by line, including its quirks:

- dictionary terms are lower-cased while documents are compared as written (Java behaviour, kept);
- `getPositionWordMap` tests `map.containsValue(term)`, i.e. the term against the numeric ids: a number such as
  "18" never enters the dictionary once id 18 exists; reproduced;
- Java's `String.split("\\s+")` keeps a leading empty token, which counts in term frequencies; reproduced.

Checks against the Java output (EmoTxt's own runs):

| Component | Data | Result |
|---|---|---|
| Uni-/bigram dictionaries | 4,800 SO posts | identical terms and order (10,490 / 70,788) |
| Unigram IDF and features | 4,800 SO posts | max \|difference\| 4.4e-16, same 69,805 non-zeros |
| Bigram features | 300 posts | identical (0.0), same 4,213 non-zeros |
| WordNet features and IDF | 300 posts | max \|difference\| 2.2e-16, same non-zeros |

Run time: 2 s instead of hours. Java remains only for EmoTxt's tokenizer (~50 ms per text).

## 3. Results on the GitHub test set

Official split of Imran et al. (1,600 train / 400 test). Model choice and thresholds never used the test set.

| Emotion (test positives) | EmoTxt, SO only | EmoTxt, SO + GH | **RoBERTa, SO + GH** | RoBERTa P / R |
|---|---|---|---|---|
| Anger (68) | 0.22 | 0.39 | **0.52** | 0.47 / 0.59 |
| Joy (84) | 0.12 | 0.41 | **0.58** | 0.66 / 0.52 |
| Love (44) | 0.53 | 0.63 | **0.68** | 0.74 / 0.64 |
| Sadness (55) | 0.42 | 0.54 | **0.65** | 0.78 / 0.56 |
| Fear (40) | 0.36 | 0.44 | **0.48** | 0.39 / 0.63 |
| Surprise (65) | 0.00 | 0.55 | **0.70** | 0.67 / 0.74 |
| **Micro-F1** | 0.28 | 0.49 | **0.60** | |

Improved RoBERTa (`train_roberta_cv.py`, `train_roberta_final.py`): SO + GitHub-train + 4 augmented variants per
comment (polarity-targeted augmentation released by Imran et al., derived from GitHub-train only), 4 epochs,
thresholds from out-of-fold predictions on the 1,600 training comments (variants never leave their comment's fold).

| Emotion | 5-fold ensemble | **Final model (deployed)** | Final P / R |
|---|---|---|---|
| Anger | 0.57 | **0.55** | 0.50 / 0.60 |
| Joy | 0.67 | **0.65** | 0.69 / 0.61 |
| Love | 0.76 | **0.67** | 0.62 / 0.73 |
| Sadness | 0.67 | **0.62** | 0.64 / 0.60 |
| Fear | 0.45 | **0.52** | 0.46 / 0.60 |
| Surprise | 0.68 | **0.70** | 0.69 / 0.71 |
| **Micro-F1** | 0.63 | **0.62** | |

Anger, as in Shaver's taxonomy, includes irritation and exasperation: with the out-of-fold threshold (0.05) the
model also labels mild frustration (e.g. "same problem here, really need the dataset"), which Senti4SD rates as
neutral; explicit anger corresponds to probabilities above 0.5. Report both when anger is central.

Reference on the same test (Imran et al., ASE 2022): original EMTk 0.42; best data-augmented tools ~0.46.

**EmoTxt** (`train_emotxt.py`): dictionary, IDF and models fitted on the training data only; LIBLINEAR logistic
regression or L2-loss SVM per emotion, cost by inner CV, balanced class weights; configuration chosen by 5-fold CV
on GitHub-train (SO 0.27, GH 0.41, SO + GH 0.43).

**RoBERTa** (`train_roberta.py`): RoBERTa-base, multi-label head, trained on SO + 80% of GitHub-train; the other 20%
(320 comments) chose the epoch (4 of 6; validation micro-F1 0.567) and one threshold per emotion. lr 2e-5, 128
tokens (95% of the texts are shorter), length-bucketed batches of 16 padded to multiples of 16 (the MPS backend
otherwise runs out of memory), ~5 minutes per epoch on an M1.

## 4. Validation on natural GitHub comments (`validate_github.py`)

The MSR 2020 gold standard was annotated with the same emotions and then mapped to polarity (love, joy → positive;
anger, sadness, fear → negative). The same mapping on our predictions:

| Model | Macro-F1 | κ | F1 positive / neutral / negative |
|---|---|---|---|
| EmoTxt, SO only | 0.59 | 0.39 | 0.59 / 0.69 / 0.50 |
| EmoTxt, SO + GH | 0.62 | 0.42 | 0.63 / 0.69 / 0.54 |
| RoBERTa, SO + GH | 0.75 | 0.62 | 0.79 / 0.74 / 0.72 |
| **RoBERTa, final (deployed)** | **0.76** | **0.63** | 0.80 / 0.75 / 0.72 |

RoBERTa labels 17% of these comments with anger, 17% joy, 10% sadness, 9% love, 9% surprise, 7% fear (the corpus
is 29% negative and 28% positive).

## 5. Threats to validity

- **Moderate accuracy.** Micro-F1 0.60: emotions are *signals*; compare groups, months and people rather than
  reading absolute rates. Fear is the weakest (F1 0.48, precision 0.39).
- **Thresholds from 320 comments.** Anger's threshold (0.05) is low; on natural data anger may be over-reported
  (17% of comments). Report rates with this caveat, or recalibrate on a larger annotated sample.
- **Annotated data are not a random sample** of GitHub comments; the MSR 2020 polarity check is the closest to
  natural data.
- English only; texts are cleaned (quotes, code, mentions removed) before classification, in training and in use.

## 6. Reproduction

```bash
# EmoTxt: one Java run tokenizes the SO corpus and writes its dictionary (emotxt/training_so_anger_anger/)
cd external/emotxt && MAX_HEAP_SIZE=6000m bash train.sh -i so_anger.csv -d sc -g -e anger   # stop after the first features file
cd .. && emotions-venv/bin/python ../backend/scripts/emotions/train_emotxt.py
# RoBERTa: single model with a validation split, then 5-fold with augmented data, then the deployed final model
HF_HOME=$PWD/hf-cache toxbert-venv/bin/python ../backend/scripts/emotions/train_roberta.py
HF_HOME=$PWD/hf-cache toxbert-venv/bin/python ../backend/scripts/emotions/train_roberta_cv.py
HF_HOME=$PWD/hf-cache toxbert-venv/bin/python ../backend/scripts/emotions/train_roberta_final.py
# validation on GitHub polarity
emotions-venv/bin/python ../backend/scripts/emotions/validate_github.py roberta-final
```

Data: `external/emotxt-gold/` (SO gold, github.com/collab-uniba/EmotionDatasetMSR18), `external/gh-emotions/`
(github.com/vcu-swim-lab/SE-Emotion-Study), `external/gold/github_gold.csv` (MSR 2020). Neither emotion dataset
states a licence: cite the papers and do not redistribute them.

## 7. OpenAI models (`EMOTION_BACKEND=openai`)

Alternative to the local RoBERTa: an OpenAI model (default `gpt-6-luna`, set with `OPENAI_MODEL`) labels each text
through the Responses API (`app/tools/openai_emotions_runner.py`).

- **Texts leave the machine.** Every cleaned comment is sent to OpenAI; check that this fits the study's ethics
  approval and data-management plan before enabling it on the full sample.
- **Prompt:** Shaver's six emotions with their subordinate categories (anger includes irritation and frustration, as
  in the gold standards), one text per request, answer constrained to a JSON schema of six booleans. Optional
  labelled examples (`openai-emotions/examples.json`) come from GitHub-train only, never from validation or test.
- **Replicability:** every answer is cached in `external/openai-emotions/cache-<model>.jsonl`, keyed by model,
  prompt and text: re-runs cost nothing and the labels can be audited. Refusals and failed calls are not stored, so
  those texts stay unclassified and are retried on the next run. Report the exact model snapshot in the paper.
- **Do not mix backends:** labels from RoBERTa and from an OpenAI model are not comparable; switch backend only
  before classifying a dataset, or reset the `emotions` column of issues and comments.
- **Cost:** about 550 input and 20 output tokens per text; at the published gpt-6-luna prices ($0.10 / $0.50 per
  million tokens) roughly $6 per 100,000 texts.

Evaluation on the same data as the local models (validation first, test once):

```bash
cd external
OPENAI_API_KEY=... emotions-venv/bin/python ../backend/scripts/emotions/eval_openai.py --shots 0    # validation (320)
OPENAI_API_KEY=... emotions-venv/bin/python ../backend/scripts/emotions/eval_openai.py --shots 12   # with examples
OPENAI_API_KEY=... emotions-venv/bin/python ../backend/scripts/emotions/eval_openai.py --shots 12 --test  # once
```

Compare with RoBERTa's validation micro-F1 (0.567) before looking at the test set. Results are appended to
`external/openai-emotions/results.jsonl`.
