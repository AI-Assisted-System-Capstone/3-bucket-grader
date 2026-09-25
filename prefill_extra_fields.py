"""
Extra pre-fill targets from the sponsor's Approach 1 deck, alongside the
harm score. Same leakage-safe input, same split, same saved TF-IDF
vectorizer as severity_model.py -- these are additional outputs bolted onto
the existing pipeline, not a new one.

Report Type is deliberately NOT modeled as a pre-fill target, as of
2026-09-25. Confirmed directly (not inferred): the analyst/reviewer fills
this field in AFTER reviewing the report, using information not available
at submission time -- the same post-review timing that makes
manager_comments/unit_actions_taken unsafe inputs to the severity model.
Predicting it from intake-only text would score well (99.3% agreement with
the true hurt label, from an earlier check) for the wrong reason: it would
just be re-deriving a hindsight label through a proxy, not adding real
pre-fill value. Kept only as a diagnostic (`check_report_type_as_proxy`,
below) to document the finding -- not trained or saved as production.

Level of Invet is deliberately NOT modeled either, for an unrelated reason:
78,475 / 80,000 rows (98%) are the single value "Pt. Safety Department
Review". A majority-class guess already gets ~98% -- there's no real
modeling problem here, and building a classifier for it would be theater,
not value. Said explicitly rather than silently skipped.
"""

import joblib
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score

from data_pipeline import train_val_test, REPORT_TYPES, HPI_DESIGNATIONS, UNLABELED

MODEL_DIR = "./severity_model"


def train_and_eval(train_df, test_df, label_col, label_names, task_name):
    train_mask = (train_df[label_col] != UNLABELED).to_numpy()
    test_mask = (test_df[label_col] != UNLABELED).to_numpy()

    vectorizer = joblib.load(f"{MODEL_DIR}/vectorizer.joblib")  # reuse, don't refit
    X_train = vectorizer.transform(train_df["text"][train_mask])
    X_test = vectorizer.transform(test_df["text"][test_mask])
    y_train = train_df[label_col][train_mask]
    y_test = test_df[label_col][test_mask]

    dummy = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    dummy_preds = dummy.predict(X_test)
    dummy_acc = accuracy_score(y_test, dummy_preds)
    dummy_f1 = f1_score(y_test, dummy_preds, average="macro", zero_division=0)

    clf = LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=-1)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    macro_f1 = f1_score(y_test, preds, average="macro", zero_division=0)

    print(f"\n=== {task_name} ===")
    print(f"Usable rows: train={train_mask.sum()}/{len(train_df)}  test={test_mask.sum()}/{len(test_df)}")
    print(f"Majority-class baseline: accuracy={dummy_acc:.4f}  macro F1={dummy_f1:.4f}")
    print(f"This model:              accuracy={acc:.4f}  macro F1={macro_f1:.4f}")
    present = sorted(set(y_test) | set(preds))
    print(classification_report(y_test, preds, labels=present,
                                 target_names=[label_names[i] for i in present], zero_division=0))
    return clf, {"accuracy": acc, "macro_f1": macro_f1,
                 "baseline_accuracy": dummy_acc, "baseline_macro_f1": dummy_f1}


def check_report_type_as_proxy(train_df):
    """Documents why Report Type is out of scope as a pre-fill target.
    CONFIRMED 2026-09-25: the analyst/reviewer sets this field after
    reviewing the report, not the reporter at intake -- so a classifier
    trained on submission-time text would be re-deriving a hindsight label,
    the same circularity problem the harm-score leakage had. This function
    is diagnostic only: it measures how strong that proxy relationship is,
    it does not train or save anything.
    """
    from data_pipeline import REPORT_TYPE2ID
    labeled = train_df[train_df["report_type_label"] != UNLABELED]
    is_serious_event = labeled["report_type_label"] == REPORT_TYPE2ID["Serious Event"]
    overlap = (is_serious_event == labeled["hurt_label"].astype(bool)).mean()
    precision_as_hurt_predictor = labeled.loc[is_serious_event, "hurt_label"].mean()
    recall_as_hurt_predictor = labeled.loc[labeled["hurt_label"] == 1, "report_type_label"].eq(
        REPORT_TYPE2ID["Serious Event"]).mean()
    print("\n=== Report Type is a post-review field -- excluded from pre-fill scope ===")
    print(f"Agreement between 'Serious Event' and true hurt label: {overlap:.4f}")
    print(f"If we used 'Serious Event' AS a hurt predictor: precision={precision_as_hurt_predictor:.4f}, "
          f"recall={recall_as_hurt_predictor:.4f}")
    print("CONFIRMED: Report Type is assigned by the analyst/reviewer after seeing the report, "
          "not at intake. A model predicting it from submission-time text would just be "
          "re-deriving that hindsight judgment through a proxy -- not modeled, no artifact saved.")


def main():
    train_df, val_df, test_df = train_val_test()

    print("Report Type: NOT modeled as a pre-fill target. Confirmed 2026-09-25: assigned by "
          "the analyst/reviewer after the report is reviewed, not by the reporter at intake -- "
          "the same post-review timing that makes manager_comments/unit_actions_taken unsafe "
          "inputs to the severity model. See check_report_type_as_proxy() below for the "
          "measured proxy relationship. No model trained, no artifact saved.")

    print("\nLevel of Invet: NOT modeled. 78,475/80,000 rows (98%) are the single value "
          "'Pt. Safety Department Review'. The remaining 2% (~1,525 rows total, including "
          "222 RCA -- Root Cause Analysis, the rare-but-most-consequential category) are too "
          "sparse in this synthetic dataset to model reliably, particularly split three ways "
          "across train/val/test. This is a data-sparsity limitation, not a claim that the "
          "field itself is unimportant -- RCA cases are exactly the ones a sponsor would care "
          "most about flagging correctly.")

    hpi_clf, hpi_metrics = train_and_eval(train_df, test_df, "hpi_label", HPI_DESIGNATIONS, "HPI Designation")
    check_report_type_as_proxy(train_df)

    joblib.dump(hpi_clf, f"{MODEL_DIR}/hpi_model.joblib")
    print(f"\nSaved hpi_model.joblib to {MODEL_DIR}/ (Report Type intentionally not saved -- see above)")


if __name__ == "__main__":
    main()
