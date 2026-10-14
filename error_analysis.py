"""
Error analysis on the saved severity model (severity_model.py): which
true-harm reports (grades E..I) does it miss on the October test set, and is
there a pattern?

A "miss" here is a harmed report that is NOT flagged for priority review:
its raw triage score falls below the frozen cutoff in severity_model/
config.json. That is the production flagging mechanism, so these are the
reports a reviewer working the flagged queue would not see. The derived
3-bucket label is a secondary summary, so harm reports it labels "none" are
counted separately at the end, for comparison with the earlier 3-bucket
models this script was first written for.

Loads the saved model rather than retraining, so run severity_model.py
first. Uses the same leakage-safe input and date split as everything else
(data_pipeline.py), so this is the same test set the reported metrics came
from.

Everything here is measured on the test set. It describes what the shipped
model misses; it is not a basis for choosing a new cutoff.

Run: python3 error_analysis.py
"""

import numpy as np
import pandas as pd

from data_pipeline import FULL_LABELS, LABELS, train_val_test
from predict import load_model
from severity_model import OUT_DIR, bucket_probs, triage_score

OUT_PATH = f"{OUT_DIR}/false_negatives.csv"  # *.csv is gitignored: contains report text
N_SAMPLES = 15


def letter(grade_label):
    return grade_label.split("-", 1)[0]


def main():
    vectorizer, clf, calibrator, cutoff = load_model()
    _, _, test_df = train_val_test()
    print("Test set size:", len(test_df))

    proba = clf.predict_proba(vectorizer.transform(test_df["text"]))
    bucket_proba = bucket_probs(proba, clf.classes_)
    triage_raw = triage_score(proba, clf.classes_)

    test_df["true_grade"] = test_df["grade"].map(letter)
    test_df["pred_grade"] = [letter(FULL_LABELS[i]) for i in clf.classes_[proba.argmax(axis=1)]]
    test_df["pred_bucket"] = np.array(LABELS)[bucket_proba.argmax(axis=1)]
    for i, bucket in enumerate(LABELS):
        test_df[f"prob_{bucket}"] = bucket_proba[:, i]
    test_df["triage_score_raw"] = triage_raw
    test_df["triage_probability"] = calibrator.predict_proba(triage_raw.reshape(-1, 1))[:, 1]
    test_df["flagged"] = triage_raw >= cutoff
    test_df["word_count"] = test_df["text"].str.split().str.len()
    test_df["says_no_harm"] = test_df["text"].str.contains("no harm", case=False)

    harmed = test_df[test_df["hurt_label"] == 1]
    fn = harmed[~harmed["flagged"]].sort_values("triage_score_raw")
    caught = harmed[harmed["flagged"]]

    print(f"\nTriage cutoff (raw score): {cutoff:.4f}")
    print(f"Harmed reports (E..I) not flagged: {len(fn)} of {len(harmed)} "
          f"(recall {len(caught) / len(harmed):.1%})")

    print("\n--- Misses by true grade ---")
    by_grade = pd.DataFrame({
        "harmed": harmed["true_grade"].value_counts(),
        "missed": fn["true_grade"].value_counts(),
    }).fillna(0).astype(int).sort_index()
    by_grade["missed_share"] = (by_grade["missed"] / by_grade["harmed"]).round(3)
    print(by_grade.to_string())

    print("\n--- Misses by true bucket ---")
    for bucket in ("some", "serious"):
        n = (harmed["bucket"] == bucket).sum()
        print(f"  {bucket:8s} {(fn['bucket'] == bucket).sum()} of {n} missed")

    if len(fn) == 0:
        print("\nNo misses at this cutoff.")
        return

    print("\n--- What the model called the misses (predicted grade) ---")
    print(fn["pred_grade"].value_counts().sort_index().to_string())

    print("\n--- How far below the cutoff? (raw triage score of the misses) ---")
    print(fn["triage_score_raw"].describe().round(3).to_string())
    near = (fn["triage_score_raw"] >= cutoff / 2).sum()
    print(f"Within half the cutoff (score >= {cutoff / 2:.3f}): {near} of {len(fn)}")

    print("\n--- Word count: misses vs. caught harm reports ---")
    print(f"Missed median word count: {fn['word_count'].median():.0f}  mean: {fn['word_count'].mean():.1f}")
    print(f"Caught median word count: {caught['word_count'].median():.0f}  mean: {caught['word_count'].mean():.1f}")

    print('\n--- Reports whose text contains "no harm" ---')
    print(f"Missed: {fn['says_no_harm'].sum()} of {len(fn)} ({fn['says_no_harm'].mean():.1%})")
    print(f"Caught: {caught['says_no_harm'].sum()} of {len(caught)} ({caught['says_no_harm'].mean():.1%})")

    # Serious misses first (there are few, and they matter most), then the
    # lowest-scoring of the rest.
    serious_fn = fn[fn["bucket"] == "serious"]
    other_fn = fn[fn["bucket"] != "serious"].head(max(N_SAMPLES - len(serious_fn), 0))
    print(f"\n--- Missed reports: all {len(serious_fn)} serious, then the "
          f"{len(other_fn)} lowest-scoring others ---")
    for _, row in pd.concat([serious_fn, other_fn]).iterrows():
        print(f"\n[{row['event_no']}  true={row['true_grade']} pred={row['pred_grade']}  "
              f"triage={row['triage_score_raw']:.3f} (calibrated {row['triage_probability']:.3f})  "
              f"words={row['word_count']}]")
        print(row["text"][:400])

    bucket_fn = harmed[harmed["pred_bucket"] == "none"]
    print("\n--- For comparison: harm reports the derived 3-bucket label calls 'none' ---")
    print(f"{len(bucket_fn)} of {len(harmed)}  "
          f"(some: {(bucket_fn['bucket'] == 'some').sum()}, serious: {(bucket_fn['bucket'] == 'serious').sum()})")
    print(f"Of those, still flagged by the triage cutoff: {bucket_fn['flagged'].sum()}")

    cols = ["event_no", "true_grade", "pred_grade", "bucket", "pred_bucket",
            "prob_none", "prob_some", "prob_serious",
            "triage_score_raw", "triage_probability", "word_count", "says_no_harm", "text"]
    fn[cols].to_csv(OUT_PATH, index=False)
    print(f"\nSaved all {len(fn)} misses to {OUT_PATH}")


if __name__ == "__main__":
    main()
