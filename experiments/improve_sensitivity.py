"""
Can a cheap change to the production model raise sensitivity at the
sponsor's operating point (96% specificity on harmed vs. not harmed)?

Each candidate is scored on the rolling folds (rolling_folds in
data_pipeline.py) whose test month is June-September: train on every earlier
month, score the test month. October is NOT used here -- it stays the final
check for whichever candidate wins. Early folds (Mar-May) are skipped because
they train on far less data than production.

Metric, per test month: the cutoff that leaves 96% of that month's unharmed
reports unflagged (cutoff_for_specificity, as in severity_model_extended.py),
and the share of harmed (E-I) reports at or above it. Pooled = caught / harmed
summed over the four months. Also reported: the same for serious (G-I) only,
and mean AUROC.

Run from the repo root:
    python3 -m experiments.improve_sensitivity
    python3 -m experiments.improve_sensitivity --october "word + char n-grams, C=4"

The second form is the one-time final check: it trains the production model
and the named winner on January-August and scores October with the same
metric. Run it once, after picking the winner on June-September.

Writes experiments/improve_sensitivity_results.json and, with --october,
experiments/improve_sensitivity_october.json (aggregates only).
"""

import argparse
import json
import os
import time

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from data_pipeline import load_dataset, rolling_folds, train_val_test
from severity_model import fit_model, make_vectorizer, triage_score
from severity_model_extended import SPONSOR_SPECIFICITY, cutoff_for_specificity

FIRST_TEST_MONTH = "2024-06"
LAST_TEST_MONTH = "2024-09"  # October deliberately excluded
OUT = "./experiments/improve_sensitivity_results.json"
OUT_OCTOBER = "./experiments/improve_sensitivity_october.json"


def word_vec(**overrides):
    v = make_vectorizer()
    v.set_params(**overrides)
    return v


def char_vec():
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                           max_features=50000, sublinear_tf=True)


class Features:
    """One or more vectorizers, horizontally stacked."""

    def __init__(self, *vecs):
        self.vecs = vecs

    def fit_transform(self, text):
        return sp.hstack([v.fit_transform(text) for v in self.vecs]).tocsr()

    def transform(self, text):
        return sp.hstack([v.transform(text) for v in self.vecs]).tocsr()


def grade_model(features, C=1.0, class_weight="balanced"):
    """Production shape: 10-grade LogReg, triage = summed E..I probability."""
    def run(train_df, test_df):
        f = features()
        clf = LogisticRegression(max_iter=2000, C=C, class_weight=class_weight)
        clf.fit(f.fit_transform(train_df["text"]), train_df["full_label"])
        return triage_score(clf.predict_proba(f.transform(test_df["text"])), clf.classes_)
    return run


def production(train_df, test_df):
    """Exactly what severity_model.py trains."""
    vectorizer, clf = fit_model(train_df)
    return triage_score(clf.predict_proba(vectorizer.transform(test_df["text"])), clf.classes_)


def hurt_model(features, C=1.0):
    """A model trained only on harmed vs. not harmed."""
    def run(train_df, test_df):
        f = features()
        clf = LogisticRegression(max_iter=2000, C=C, class_weight="balanced")
        clf.fit(f.fit_transform(train_df["text"]), train_df["hurt_label"])
        return clf.predict_proba(f.transform(test_df["text"]))[:, 1]
    return run


def blend(*runs):
    """Average of each model's within-month rank, so scales don't matter."""
    def run(train_df, test_df):
        ranks = [np.argsort(np.argsort(r(train_df, test_df))) / len(test_df) for r in runs]
        return np.mean(ranks, axis=0)
    return run


WORD = lambda: Features(make_vectorizer())
WORD_BIG = lambda: Features(word_vec(sublinear_tf=True, max_features=100000, min_df=2))
WORD_CHAR = lambda: Features(make_vectorizer(), char_vec())

CANDIDATES = {
    "production": production,
    "C=0.25": grade_model(WORD, C=0.25),
    "C=4": grade_model(WORD, C=4.0),
    "no class weights": grade_model(WORD, class_weight=None),
    "bigger vocab, sublinear tf": grade_model(WORD_BIG),
    "word + char n-grams": grade_model(WORD_CHAR),
    "binary hurt model": hurt_model(WORD),
    "blend: production + binary hurt": blend(production, hurt_model(WORD)),
    "C=16": grade_model(WORD, C=16.0),
    "word + char n-grams, C=4": grade_model(WORD_CHAR, C=4.0),
    "word + char n-grams, C=16": grade_model(WORD_CHAR, C=16.0),
}


def score_month(y_hurt, y_serious, scores):
    cut = cutoff_for_specificity(y_hurt, scores, SPONSOR_SPECIFICITY)
    flagged = scores >= cut
    return {
        "hurt_caught": int((flagged & (y_hurt == 1)).sum()), "hurt_n": int(y_hurt.sum()),
        "serious_caught": int((flagged & y_serious).sum()), "serious_n": int(y_serious.sum()),
        "auroc": float(roc_auc_score(y_hurt, scores)),
    }


def october_check(winner):
    train_df, _, test_df = train_val_test()
    out = {}
    for name in ["production", winner]:
        r = score_month(test_df["hurt_label"].to_numpy(), (test_df["bucket"] == "serious").to_numpy(),
                        CANDIDATES[name](train_df, test_df))
        r["sensitivity"] = r["hurt_caught"] / r["hurt_n"]
        out[name] = r
        print(f"{name:34s} October sens@96spec {r['sensitivity']:.4f} ({r['hurt_caught']}/{r['hurt_n']})  "
              f"serious {r['serious_caught']}/{r['serious_n']}", flush=True)
    with open(OUT_OCTOBER, "w") as f:
        json.dump(out, f, indent=2)
    print("Saved", OUT_OCTOBER)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="+", help="run just these candidates; results merge into the JSON")
    parser.add_argument("--october", metavar="WINNER",
                        help="final check: score production and WINNER once on October")
    args = parser.parse_args()
    if args.october:
        return october_check(args.october)
    names = args.only or list(CANDIDATES)

    df = load_dataset()
    folds = [(tr, te) for tr, _, te in rolling_folds(df)
             if FIRST_TEST_MONTH <= te["event_month"].iloc[0] <= LAST_TEST_MONTH]
    print("Test months:", [te["event_month"].iloc[0] for _, te in folds])

    results = {}
    if args.only and os.path.exists(OUT):
        results = json.load(open(OUT))["candidates"]
    for name in names:
        run = CANDIDATES[name]
        t0 = time.time()
        months = {}
        for train_df, test_df in folds:
            # Cutoff month is not needed: the metric sets its own cutoff from
            # the test month's unharmed reports, like the sponsor's number.
            scores = run(train_df, test_df)
            months[test_df["event_month"].iloc[0]] = score_month(
                test_df["hurt_label"].to_numpy(), (test_df["bucket"] == "serious").to_numpy(), scores)
        caught = sum(m["hurt_caught"] for m in months.values())
        n = sum(m["hurt_n"] for m in months.values())
        s_caught = sum(m["serious_caught"] for m in months.values())
        s_n = sum(m["serious_n"] for m in months.values())
        results[name] = {
            "pooled_sensitivity_at_96_specificity": caught / n,
            "pooled_counts": f"{caught}/{n}",
            "serious_caught_at_96_specificity": f"{s_caught}/{s_n}",
            "mean_auroc": float(np.mean([m["auroc"] for m in months.values()])),
            "by_month": months,
        }
        r = results[name]
        print(f"{name:34s} sens@96spec {r['pooled_sensitivity_at_96_specificity']:.4f} ({r['pooled_counts']})  "
              f"serious {r['serious_caught_at_96_specificity']}  AUROC {r['mean_auroc']:.4f}  "
              f"Sep {months['2024-09']['hurt_caught']}/{months['2024-09']['hurt_n']}  ({time.time() - t0:.0f}s)",
              flush=True)

    with open(OUT, "w") as f:
        json.dump({"config": {"test_months": f"{FIRST_TEST_MONTH}..{LAST_TEST_MONTH}",
                              "specificity": SPONSOR_SPECIFICITY, "october_used": False},
                   "candidates": results}, f, indent=2)
    print("Saved", OUT)


if __name__ == "__main__":
    main()
