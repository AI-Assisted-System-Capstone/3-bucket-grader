"""
Two more pre-fill targets from the sponsor's Approach 1 deck, alongside the
harm score: Report Type and HPI Designation. Same leakage-safe input, same
split, same saved TF-IDF vectorizer as severity_model.py -- these are
additional outputs bolted onto the existing pipeline, not a new one.

Level of Invet is deliberately NOT modeled: 78,475 / 80,000 rows (98%) are
the single value "Pt. Safety Department Review". A majority-class guess
already gets ~98% -- there's no real modeling problem here, and building a
classifier for it would be theater, not value. Said explicitly rather than
silently skipped.
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
    """Does Report Type == 'Serious Event' just restate the E-I hurt label?
    If so, pre-filling it from the model is close to circular -- the sponsor
    needs to confirm whether this field is set AT INTAKE (worth pre-filling)
    or assigned by the reviewer AFTER seeing the harm outcome (same
    leakage/circularity problem as the harm score itself, just one field
    over). This is an empirical check on overlap, not an answer to that
    question -- only the sponsor knows how the field is actually populated.
    """
    from data_pipeline import REPORT_TYPE2ID
    labeled = train_df[train_df["report_type_label"] != UNLABELED]
    is_serious_event = labeled["report_type_label"] == REPORT_TYPE2ID["Serious Event"]
    overlap = (is_serious_event == labeled["hurt_label"].astype(bool)).mean()
    precision_as_hurt_predictor = labeled.loc[is_serious_event, "hurt_label"].mean()
    recall_as_hurt_predictor = labeled.loc[labeled["hurt_label"] == 1, "report_type_label"].eq(
        REPORT_TYPE2ID["Serious Event"]).mean()
    print("\n=== Is 'Report Type = Serious Event' just a proxy for hurt (E-I)? ===")
    print(f"Agreement between 'Serious Event' and true hurt label: {overlap:.4f}")
    print(f"If we used 'Serious Event' AS a hurt predictor: precision={precision_as_hurt_predictor:.4f}, "
          f"recall={recall_as_hurt_predictor:.4f}")
    print("NEEDS SPONSOR CONFIRMATION: is Report Type known at intake, or assigned after review? "
          "If the latter, pre-filling it from a model trained on this label has the same "
          "circularity problem the harm score leakage had.")


def main():
    train_df, val_df, test_df = train_val_test()

    print("Level of Invet: NOT modeled. 78,475/80,000 rows (98%) are the single value "
          "'Pt. Safety Department Review'. The remaining 2% (~1,525 rows total, including "
          "222 RCA -- Root Cause Analysis, the rare-but-most-consequential category) are too "
          "sparse in this synthetic dataset to model reliably, particularly split three ways "
          "across train/val/test. This is a data-sparsity limitation, not a claim that the "
          "field itself is unimportant -- RCA cases are exactly the ones a sponsor would care "
          "most about flagging correctly.")

    report_type_clf, report_type_metrics = train_and_eval(train_df, test_df, "report_type_label", REPORT_TYPES, "Report Type")
    hpi_clf, hpi_metrics = train_and_eval(train_df, test_df, "hpi_label", HPI_DESIGNATIONS, "HPI Designation")
    check_report_type_as_proxy(train_df)

    joblib.dump(report_type_clf, f"{MODEL_DIR}/report_type_model.joblib")
    joblib.dump(hpi_clf, f"{MODEL_DIR}/hpi_model.joblib")
    print(f"\nSaved report_type_model.joblib and hpi_model.joblib to {MODEL_DIR}/")


if __name__ == "__main__":
    main()
