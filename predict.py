"""
Score new reports with the saved severity model (severity_model.py).

Loads the vectorizer, model, calibrator and frozen triage cutoff from
severity_model/ and scores every row of a CSV or parquet file. Input text is
built by data_pipeline.build_input_text, the same function used in training,
so only the leakage-safe intake fields are read -- post-investigation columns
may be present in the file but are never passed to the model.

Output, one row per input row, same order:
  event_no            copied from "Event No." when the input has it
  grade               predicted PSRS letter (A, B1, B2, C, D, E, F, G, H, I)
  grade_label         the same grade, full name
  bucket              none / some / serious, derived from the grade probabilities
  triage_score_raw    summed probability of grades E..I (what the cutoff applies to)
  triage_probability  calibrated probability that the patient was harmed
  flagged             True if triage_score_raw >= the frozen cutoff

Rows with no input text at all (no narrative and no intake fields) are not
scored: training drops them, so the model has nothing to say about them.
They stay in the output with empty prediction columns.

Run: python3 predict.py reports.csv [-o predictions.csv]
"""

import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd

from data_pipeline import FEATURE_COLUMNS, FULL_LABELS, ID_COL, LABELS, build_input_text
from severity_model import OUT_DIR, bucket_probs, triage_score

PARQUET_EXTENSIONS = (".parquet", ".pq")


def read_reports(path):
    if path.lower().endswith(PARQUET_EXTENSIONS):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def write_predictions(df, path):
    if path.lower().endswith(PARQUET_EXTENSIONS):
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def load_model(model_dir=OUT_DIR):
    vectorizer = joblib.load(f"{model_dir}/vectorizer.joblib")
    clf = joblib.load(f"{model_dir}/model.joblib")
    calibrator = joblib.load(f"{model_dir}/calibrator.joblib")
    with open(f"{model_dir}/config.json") as f:
        cutoff = json.load(f)["cutoff_raw_triage"]
    return vectorizer, clf, calibrator, cutoff


def score_reports(reports, model_dir=OUT_DIR):
    """Score a DataFrame of raw reports. Needs the FEATURE_COLUMNS; any other
    columns are ignored."""
    vectorizer, clf, calibrator, cutoff = load_model(model_dir)

    # Select the feature columns explicitly so excluded (post-investigation)
    # columns in the raw file never reach build_input_text. It raises
    # DataContractError itself if a feature column is missing.
    present = [c for c in FEATURE_COLUMNS if c in reports.columns]
    text = build_input_text(reports[present])
    has_text = (text.str.len() > 0).to_numpy()

    out = pd.DataFrame(index=reports.index)
    if ID_COL in reports.columns:
        out["event_no"] = reports[ID_COL]
    out["grade"] = None
    out["grade_label"] = None
    out["bucket"] = None
    out["triage_score_raw"] = np.nan
    out["triage_probability"] = np.nan
    out["flagged"] = None

    if has_text.any():
        proba = clf.predict_proba(vectorizer.transform(text[has_text]))
        grade_labels = [FULL_LABELS[i] for i in clf.classes_[proba.argmax(axis=1)]]
        triage_raw = triage_score(proba, clf.classes_)

        out.loc[has_text, "grade"] = [g.split("-", 1)[0] for g in grade_labels]
        out.loc[has_text, "grade_label"] = grade_labels
        out.loc[has_text, "bucket"] = np.array(LABELS)[bucket_probs(proba, clf.classes_).argmax(axis=1)]
        out.loc[has_text, "triage_score_raw"] = triage_raw
        out.loc[has_text, "triage_probability"] = calibrator.predict_proba(triage_raw.reshape(-1, 1))[:, 1]
        out.loc[has_text, "flagged"] = triage_raw >= cutoff

    return out


def main():
    parser = argparse.ArgumentParser(description="Score reports with the saved severity model.")
    parser.add_argument("input", help="CSV or parquet file of reports")
    parser.add_argument("-o", "--output", help="CSV or parquet file to write "
                        "(default: <input name>_predictions.csv in the current directory)")
    parser.add_argument("--model-dir", default=OUT_DIR, help=f"saved model directory (default: {OUT_DIR})")
    args = parser.parse_args()

    output = args.output
    if output is None:
        stem = os.path.splitext(os.path.basename(args.input))[0]
        output = f"./{stem}_predictions.csv"

    reports = read_reports(args.input)
    predictions = score_reports(reports, args.model_dir)
    write_predictions(predictions, output)

    scored = predictions["grade"].notna()
    n_scored = int(scored.sum())
    n_flagged = int(predictions.loc[scored, "flagged"].astype(bool).sum())
    print(f"Scored {n_scored} of {len(predictions)} reports "
          f"({len(predictions) - n_scored} skipped, no input text).")
    if n_scored:
        print(f"Flagged for priority review: {n_flagged} ({n_flagged / n_scored:.1%})")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
