"""
Error analysis: which true-harm reports (grades E-I) does the production
severity model miss, and is there a pattern?

Uses the same leakage-safe input and date-based split as severity_model.py
(data_pipeline.py), so this looks at the same held-out October test set the
reported metrics came from. Loads the saved model from severity_model/ through
predict.load_model, and applies the triage cutoff frozen on validation --
nothing is re-tuned here.

A "miss" is a harmed patient (E-I) whose raw triage score falls below the
cutoff, i.e. a report that would not be flagged for priority review.

Run from the repo root (after `python3 severity_model.py` has saved the model):
    python3 -m experiments.error_analysis
"""

import os

import numpy as np
import pandas as pd

from data_pipeline import train_val_test, LABELS, FULL_LABELS
from predict import load_model
from severity_model import OUT_DIR, bucket_probs, triage_score

OUT_CSV = os.path.join(OUT_DIR, "false_negatives.csv")  # *.csv is gitignored


def main():
    _, _, test_df = train_val_test()
    test_df = test_df.copy()
    print("Test set size:", len(test_df))

    model = load_model(OUT_DIR)
    proba = model.clf.predict_proba(model.vectorizer.transform(test_df["text"]))
    classes = model.clf.classes_
    bucket_p = bucket_probs(proba, classes)
    triage_raw = triage_score(proba, classes)

    test_df["pred_grade"] = [FULL_LABELS[c] for c in classes[proba.argmax(axis=1)]]
    test_df["pred_bucket"] = np.array(LABELS)[bucket_p.argmax(axis=1)]
    for i, b in enumerate(LABELS):
        test_df[f"p_{b}"] = bucket_p[:, i]
    test_df["triage_score_raw"] = triage_raw
    test_df["flagged"] = triage_raw >= model.cutoff
    test_df["word_count"] = test_df["text"].str.split().str.len()

    hurt = test_df[test_df["hurt_label"] == 1]
    fn = hurt[~hurt["flagged"]].copy()
    caught = hurt[hurt["flagged"]]
    print(f"\nCutoff (frozen on validation): {model.cutoff:.4f}")
    print(f"Harmed patients (E-I): {len(hurt)}, flagged: {len(caught)}, missed: {len(fn)}")
    print("\nMissed, by true grade:")
    print(fn["grade"].value_counts())
    print("\nMissed, by true bucket:")
    print(fn["bucket"].value_counts())

    print("\n--- Word count: missed vs. caught harm cases ---")
    print("Missed median word count:", fn["word_count"].median(), " mean:", round(fn["word_count"].mean(), 1))
    print("Caught median word count:", caught["word_count"].median(), " mean:", round(caught["word_count"].mean(), 1))

    print("\n--- How close were the misses? (raw triage score vs. cutoff) ---")
    print(fn["triage_score_raw"].describe())
    near = (fn["triage_score_raw"] >= model.cutoff / 2).sum()
    print(f"Near-misses (score at least half the cutoff): {near}")

    print("\n--- Missed cases, lowest score first (up to 15) ---")
    for _, row in fn.sort_values("triage_score_raw").head(15).iterrows():
        print(f"\n[{row['event_no']} true={row['grade']} pred={row['pred_grade']} "
              f"score={row['triage_score_raw']:.3f} p(none)={row['p_none']:.2f} "
              f"p(some)={row['p_some']:.2f} p(serious)={row['p_serious']:.2f} words={row['word_count']}]")
        print(row["text"][:400])

    fn.to_csv(OUT_CSV, index=False)
    print(f"\nSaved all missed harm cases to {OUT_CSV}")


if __name__ == "__main__":
    main()
