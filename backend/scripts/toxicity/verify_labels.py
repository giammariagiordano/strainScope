"""ASE 2026 pipeline, verifier step: the package's random forest (trained on the 898 human-labelled conversations)
reads the eight event-category scores and estimates whether the LLM's label is correct; conversations it doubts go
to human review. Same scaler, model and 0.5 cut-off as the package's select_incorrect_data.ipynb.

Run from external/toxicity-gh with toxbert-venv (scikit-learn 1.5.2, the package's version):
    ../toxbert-venv/bin/python ../../backend/scripts/toxicity/verify_labels.py work/llm/strainscope.jsonl \
        work/strainscope_conversations.parquet
    # validation of the local setup on the human-labelled conversations:
    ../toxbert-venv/bin/python ../../backend/scripts/toxicity/verify_labels.py work/llm/imran.jsonl \
        work/imran_conversations.parquet --id-col issue_id --human label

Outputs next to the JSONL: <name>_verified.parquet (all rows + verifier probability) and <name>_review.csv (the
flagged conversations with text, LLM label and explanation, and an empty human_label column to fill in).
With --human, also prints the LLM's and the pipeline's agreement with the human labels and with the package's own
Ministral outputs, and writes <name>_validation.json.
"""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import cohen_kappa_score, precision_recall_fscore_support

VERIFIER = Path("backup/output/verifier-random-forest-model")
ASE_IMRAN = Path("backup/output/llm-outputs/ministral-3-14b/unified_final_dataset_conversations_toxicity_and_category.csv")
CATEGORIES = ["person_hostility", "process_hostility", "dismissiveness", "escalation_without_explicit_abuse",
              "subtle_criticism", "hostility_density", "resolution_language", "behavior_callout"]  # fmt: skip


def scores(y, pred) -> dict:
    p, r, f, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    return {"precision": round(p, 3), "recall": round(r, 3), "f1": round(f, 3), "accuracy": round((y == pred).mean(), 4),
            "kappa": round(cohen_kappa_score(y, pred), 3), "flagged_toxic": int(pred.sum()), "n": len(y)}  # fmt: skip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("labels")
    ap.add_argument("conversations")
    ap.add_argument("--id-col", default="id")
    ap.add_argument("--human", help="column of the conversations file with human labels (validation run)")
    args = ap.parse_args()

    llm = pd.read_json(args.labels, lines=True)
    conv = pd.read_parquet(args.conversations).rename(columns={args.id_col: "id"})
    df = llm.merge(conv, on="id", how="left")
    scaler = joblib.load(VERIFIER / "standardScaler_70_30_split_with_category_score.pkl")
    rf = joblib.load(VERIFIER / "rf_model_with_category_scores.joblib")
    X = pd.DataFrame(scaler.transform(df[CATEGORIES]), columns=CATEGORIES)
    df["verifier_p_correct"] = rf.predict_proba(X)[:, 1]
    df["needs_review"] = df.verifier_p_correct < 0.5

    stem = Path(args.labels).with_suffix("")
    df.to_parquet(f"{stem}_verified.parquet")
    review = df[df.needs_review].sort_values("verifier_p_correct")
    review.assign(human_label="")[["id", "repo", "number", "toxicity_prediction", "verifier_p_correct", "explanation_text",
                                   "text", "human_label"] if "repo" in df else
                                  ["id", "toxicity_prediction", "verifier_p_correct", "explanation_text", "text",
                                   "human_label"]].to_csv(f"{stem}_review.csv", index=False)  # fmt: skip

    n = len(df)
    print(f"{n} conversations, LLM toxic {df.toxicity_prediction.sum()} ({df.toxicity_prediction.mean():.2%}), "
          f"truncated {df.truncated.sum()}")  # fmt: skip
    print(f"verifier flags {len(review)} ({len(review) / n:.1%}) for review, of which LLM-toxic "
          f"{review.toxicity_prediction.sum()} -> {stem}_review.csv")  # fmt: skip

    if args.human:
        y = df[args.human].astype(int)
        ase = pd.read_csv(ASE_IMRAN).rename(columns={"issue_id": "id"})[["id", "toxicity_prediction"]]
        both = df.merge(ase, on="id", suffixes=("", "_ase")).dropna(subset=["toxicity_prediction_ase"])
        # pipeline after review, simulated: humans fix exactly the flagged conversations
        reviewed = df.toxicity_prediction.where(~df.needs_review, y)
        result = {
            "llm_vs_human": scores(y, df.toxicity_prediction),
            "package_llm_vs_human": scores(both[args.human].astype(int), both.toxicity_prediction_ase.astype(int)),
            "llm_vs_package_llm": scores(both.toxicity_prediction_ase.astype(int), both.toxicity_prediction),
            "pipeline_after_review_vs_human": scores(y, reviewed),
            "flagged_for_review": int(df.needs_review.sum()),
        }
        print(json.dumps(result, indent=1))
        Path(f"{stem}_validation.json").write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
