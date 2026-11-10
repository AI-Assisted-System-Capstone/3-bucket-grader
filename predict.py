"""
Score new incident reports with the saved severity model.

Loads the artifacts severity_model.py writes to severity_model/ (vectorizer,
10-grade classifier, Platt calibrator, frozen triage cutoff) and scores every
row of a CSV or parquet file:

    python3 predict.py reports.parquet -o scored.csv
    python3 predict.py reports.csv -o scored.parquet --model-dir ./severity_model
    python3 predict.py reports.parquet -o scored.csv --model distilbert

`--model distilbert` loads a fine-tuned 10-grade DistilBERT from
distilbert_output/grade_len256/final (override with --model-dir); see
distilbert_model.py. The output columns are the same for both models.

Output, one row per input row, in input order:
    event_no            passed through from `Event No.` if the input has it
    grade               predicted PSRS letter grade (A..I)
    bucket              derived none / some / serious
    p_none, p_some, p_serious   bucket probabilities (sum to 1)
    triage_score_raw    summed E..I probability (what the cutoff applies to)
    triage_probability  calibrated probability the patient was harmed
    flagged             triage_score_raw >= the cutoff frozen on validation

Only the leakage-safe intake fields in data_pipeline.FEATURE_COLUMNS are read
from the input. Post-investigation columns (manager_comments, ...) may be
present in an export -- they are dropped before the model sees anything, and
data_pipeline.build_input_text asserts that. The narrative text is not
copied into the output.
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd

from data_pipeline import FEATURE_COLUMNS, FULL_LABELS, ID_COL, LABELS, TEXT_COL, build_input_text
from severity_model import OUT_DIR, bucket_probs, triage_score


@dataclass
class SeverityModel:
    vectorizer: object
    clf: object
    calibrator: object
    cutoff: float

    @property
    def classes_(self):
        return self.clf.classes_

    def predict_proba(self, texts):
        return self.clf.predict_proba(self.vectorizer.transform(texts))


MODELS = ("tfidf", "distilbert")
DEFAULT_DISTILBERT_DIR = "./distilbert_output/grade_len256/final"


def load_model(model_dir=OUT_DIR):
    paths = {name: os.path.join(model_dir, f"{name}.joblib") for name in ("vectorizer", "model", "calibrator")}
    missing = [p for p in list(paths.values()) + [os.path.join(model_dir, "config.json")] if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            f"Model artifacts missing: {missing}. The .joblib files are gitignored -- "
            f"run `python3 severity_model.py` first to train and save them."
        )
    with open(os.path.join(model_dir, "config.json")) as f:
        cfg = json.load(f)
    return SeverityModel(
        vectorizer=joblib.load(paths["vectorizer"]),
        clf=joblib.load(paths["model"]),
        calibrator=joblib.load(paths["calibrator"]),
        cutoff=float(cfg["cutoff_raw_triage"]),
    )


def load_any_model(kind="tfidf", model_dir=None):
    if kind == "tfidf":
        return load_model(model_dir or OUT_DIR)
    if kind == "distilbert":
        from distilbert_model import load_distilbert
        return load_distilbert(model_dir or DEFAULT_DISTILBERT_DIR)
    raise ValueError(f"Unknown model {kind!r}; expected one of {MODELS}")


def read_reports(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".parquet":
        return pd.read_parquet(path)
    if ext == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported input type {ext!r}; expected .csv or .parquet")


def write_scores(scores, path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".parquet":
        scores.to_parquet(path, index=False)
    elif ext == ".csv":
        scores.to_csv(path, index=False)
    else:
        raise ValueError(f"Unsupported output type {ext!r}; expected .csv or .parquet")


def score_reports(reports, model):
    """Score a DataFrame of raw reports with either model. Returns one output row per input row."""
    if TEXT_COL not in reports.columns:
        raise ValueError(f"Input has no {TEXT_COL!r} column -- nothing to score")
    # Optional intake fields that are absent are treated as blank, the same as
    # a missing value in a row; say so, since it changes the model's input.
    absent = [c for c in FEATURE_COLUMNS if c not in reports.columns]
    if absent:
        print(f"[predict] Input lacks intake columns {absent}; treating them as blank.", file=sys.stderr)
    features = reports.reindex(columns=list(FEATURE_COLUMNS))

    proba = model.predict_proba(build_input_text(features))
    classes = model.classes_

    bucket_p = bucket_probs(proba, classes)
    triage_raw = triage_score(proba, classes)
    out = pd.DataFrame({
        "grade": [FULL_LABELS[c] for c in classes[proba.argmax(axis=1)]],
        "bucket": np.array(LABELS)[bucket_p.argmax(axis=1)],
        **{f"p_{b}": bucket_p[:, i] for i, b in enumerate(LABELS)},
        "triage_score_raw": triage_raw,
        "triage_probability": model.calibrator.predict_proba(triage_raw.reshape(-1, 1))[:, 1],
        "flagged": triage_raw >= model.cutoff,
    })
    if ID_COL in reports.columns:
        out.insert(0, "event_no", reports[ID_COL].to_numpy())
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description="Score incident reports with the saved severity model.")
    parser.add_argument("input", help="CSV or parquet file of reports")
    parser.add_argument("-o", "--output", required=True, help="where to write scores (.csv or .parquet)")
    parser.add_argument("--model", choices=MODELS, default="tfidf",
                        help="which saved model to score with (default tfidf, the production model)")
    parser.add_argument("--model-dir", default=None,
                        help=f"saved model directory (default {OUT_DIR} for tfidf, {DEFAULT_DISTILBERT_DIR} for distilbert)")
    args = parser.parse_args(argv)

    model = load_any_model(args.model, args.model_dir)
    scores = score_reports(read_reports(args.input), model)
    write_scores(scores, args.output)

    print(f"Scored {len(scores)} reports -> {args.output}")
    print(f"Flagged for priority review: {int(scores['flagged'].sum())} ({scores['flagged'].mean():.1%}) "
          f"at cutoff {model.cutoff:.4f}")
    print("Buckets:", scores["bucket"].value_counts().reindex(LABELS, fill_value=0).to_dict())


if __name__ == "__main__":
    main()
