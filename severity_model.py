"""
The finished severity model (Branch 2, the 10-grade model) -- trained, saved
to disk, and fully evaluated. This replaces baseline_grader.py /
finetune_distilbert.py (Branch 1, 3-bucket-only) as the one severity model:
everything Branch 1 produced is now derived from this one model's output,
plus the things Branch 1 could never produce (the real PSRS letter, a
calibrated triage cutoff).

What this script does, in order:

1. Train TF-IDF + multinomial Logistic Regression on the full 10-grade scale
   (A..I), same data as everything else here (data_pipeline.py: leakage-safe
   fields, real Jan-Aug/Sep/Oct split).
2. Derive three things from that one model's output:
     - letter grade (argmax over the 10 grades)
     - 3-bucket label (argmax over summed probabilities per BUCKET_GROUPS --
       replaces Branch 1)
     - triage score (sum of E..I probabilities)
3. Pick a triage cutoff on the VALIDATION set only, targeting >=95% recall on
   true hurt cases -- this project's own default target, NOT a documented
   sponsor requirement (see severity_model_extended.py for the full menu of
   alternatives, and the sponsor's own reported operating point for
   comparison) -- then freeze it and apply it, unchanged, to the test set.
4. Calibrate the triage score with Platt scaling (a 1-feature logistic
   regression mapping raw triage score -> calibrated probability), fit on
   validation, because the training set deliberately oversamples harm
   (14.4% some+serious) versus the natural rate in validation/test (~4.8%) --
   same distribution-shift problem the teammate's evaluation.py solves for.
5. Evaluate everything on the untouched October test set and save:
     - the trained vectorizer + model + cutoff + calibrator (severity_model/)
     - a full JSON evaluation report (severity_model/evaluation.json)
6. Run one example report through the saved model end to end, to prove the
   saved artifacts actually work standalone (not just in-memory objects from
   this script's own training run).

Note on "multi-seed": the teammate's neural-network heads need multiple
random seeds because their training has random initialization. TF-IDF +
Logistic Regression (lbfgs solver) has no random initialization -- the same
data always converges to the same weights -- so re-running with different
seeds here would not measure anything real. Skipped for that reason, not
skipped by oversight.
"""

import json
import os

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    mean_absolute_error,
    cohen_kappa_score,
    accuracy_score,
    classification_report,
    brier_score_loss,
)

from data_pipeline import (
    train_val_test,
    FULL_LABELS,
    FULL_LABEL2ID,
    LABELS,
    BUCKET_GROUPS,
    build_input_text,
)

OUT_DIR = "./severity_model"
TARGET_RECALL = 0.95  # sponsor's stated hard requirement on catching true harm


def bucket_probs(proba, classes_):
    """Sum a (n, n_classes) probability matrix into (n, 3) bucket probabilities,
    respecting whatever subset/order clf.classes_ actually has."""
    out = np.zeros((proba.shape[0], len(LABELS)))
    for bi, bucket in enumerate(LABELS):
        group_ids = set(BUCKET_GROUPS[bucket])
        cols = [i for i, c in enumerate(classes_) if c in group_ids]
        out[:, bi] = proba[:, cols].sum(axis=1)
    return out


def triage_score(proba, classes_):
    hurt_ids = set(BUCKET_GROUPS["some"]) | set(BUCKET_GROUPS["serious"])
    cols = [i for i, c in enumerate(classes_) if c in hurt_ids]
    return proba[:, cols].sum(axis=1)


def pick_cutoff_for_recall(y_true, scores, target_recall):
    """Highest cutoff that still achieves >= target_recall on this (validation) set."""
    order = np.argsort(scores)[::-1]
    y_sorted = y_true[order]
    s_sorted = scores[order]
    n_pos = y_true.sum()
    cum_pos = np.cumsum(y_sorted)
    recall_at_k = cum_pos / n_pos
    idx = np.searchsorted(recall_at_k, target_recall)
    idx = min(idx, len(s_sorted) - 1)
    return float(s_sorted[idx])


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    train_df, val_df, test_df = train_val_test()
    print(f"Train {train_df.shape}  Val {val_df.shape}  Test {test_df.shape}")

    # --- 1. Train ---
    # token_pattern fixed, 2026-09-25: sklearn's default (\b\w\w+\b) requires 2+
    # word characters and drops single-character tokens, so "X-ray" tokenizes
    # to "ray" and "T-cell" to "cell" -- silently wrong words shown in
    # explanations (a Safety Officer seeing "ray" loses trust in the model),
    # and the same silent mangling happens throughout training and inference,
    # not just the explanation demo.
    vectorizer = TfidfVectorizer(
        stop_words="english", max_features=20000, ngram_range=(1, 2), min_df=3,
        token_pattern=r"(?u)\b[\w-]+\b",
    )
    X_train = vectorizer.fit_transform(train_df["text"])
    X_val = vectorizer.transform(val_df["text"])
    X_test = vectorizer.transform(test_df["text"])

    clf = LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=-1)
    clf.fit(X_train, train_df["full_label"])
    print("Trained. Classes present:", [FULL_LABELS[i] for i in clf.classes_])

    # --- 2. Derive letter grade, 3-bucket, triage score, for val + test ---
    proba_val = clf.predict_proba(X_val)
    proba_test = clf.predict_proba(X_test)

    grade_pred_test = clf.predict(X_test)
    bucket_proba_test = bucket_probs(proba_test, clf.classes_)
    bucket_pred_test = np.array(LABELS)[bucket_proba_test.argmax(axis=1)]
    triage_val = triage_score(proba_val, clf.classes_)
    triage_test = triage_score(proba_test, clf.classes_)

    # --- 3. Pick triage cutoff on validation only, freeze, apply to test ---
    y_hurt_val = val_df["hurt_label"].to_numpy()
    y_hurt_test = test_df["hurt_label"].to_numpy()
    cutoff = pick_cutoff_for_recall(y_hurt_val, triage_val, TARGET_RECALL)
    flagged_test = (triage_test >= cutoff).astype(int)
    achieved_recall_val = recall_score(y_hurt_val, (triage_val >= cutoff).astype(int))

    # --- 4. Calibrate triage score (Platt scaling), fit on validation ---
    calibrator = LogisticRegression()
    calibrator.fit(triage_val.reshape(-1, 1), y_hurt_val)
    calibrated_test = calibrator.predict_proba(triage_test.reshape(-1, 1))[:, 1]
    brier_raw = brier_score_loss(y_hurt_test, triage_test)
    brier_calibrated = brier_score_loss(y_hurt_test, calibrated_test)

    # --- 5. Evaluate on test, save everything ---
    y_full_test = test_df["full_label"].to_numpy()
    y_bucket_test = test_df["bucket"].to_numpy()

    present = sorted(set(y_full_test) | set(grade_pred_test))
    grade_report = classification_report(
        y_full_test, grade_pred_test, labels=present,
        target_names=[FULL_LABELS[i] for i in present], output_dict=True, zero_division=0,
    )
    bucket_report = classification_report(
        y_bucket_test, bucket_pred_test, labels=LABELS, output_dict=True, zero_division=0,
    )

    results = {
        "config": {"target_recall": TARGET_RECALL, "cutoff_raw_triage": cutoff},
        "grade_10way": {
            "exact_match_accuracy": accuracy_score(y_full_test, grade_pred_test),
            "mae_ordinal": mean_absolute_error(y_full_test, grade_pred_test),
            "quadratic_weighted_kappa": cohen_kappa_score(y_full_test, grade_pred_test, weights="quadratic"),
            "per_grade": grade_report,
        },
        "bucket_3way_derived": {
            "note": "derived from the same 10-grade model by summing group probabilities -- replaces baseline_grader.py",
            "per_bucket": bucket_report,
        },
        "triage": {
            "auprc": average_precision_score(y_hurt_test, triage_test),
            "auroc": roc_auc_score(y_hurt_test, triage_test),
            "cutoff_chosen_on_validation": cutoff,
            "recall_on_validation_at_cutoff": achieved_recall_val,
            "test_precision_at_cutoff": precision_score(y_hurt_test, flagged_test),
            "test_recall_at_cutoff": recall_score(y_hurt_test, flagged_test),
            "test_flag_rate": float(flagged_test.mean()),
            "brier_score_raw": brier_raw,
            "brier_score_calibrated": brier_calibrated,
        },
    }

    with open(f"{OUT_DIR}/evaluation.json", "w") as f:
        json.dump(results, f, indent=2, default=float)

    joblib.dump(vectorizer, f"{OUT_DIR}/vectorizer.joblib")
    joblib.dump(clf, f"{OUT_DIR}/model.joblib")
    joblib.dump(calibrator, f"{OUT_DIR}/calibrator.joblib")
    with open(f"{OUT_DIR}/config.json", "w") as f:
        json.dump({"cutoff_raw_triage": cutoff, "target_recall": TARGET_RECALL}, f, indent=2)

    print("\n" + "=" * 70)
    print("10-GRADE MODEL, test set")
    print("=" * 70)
    print(f"Exact-letter accuracy: {results['grade_10way']['exact_match_accuracy']:.4f}")
    print(f"MAE (ordinal):         {results['grade_10way']['mae_ordinal']:.4f}")
    print(f"Quadratic kappa:       {results['grade_10way']['quadratic_weighted_kappa']:.4f}")

    print("\n" + "=" * 70)
    print("3-BUCKET, DERIVED from the 10-grade model (replaces baseline_grader.py)")
    print("=" * 70)
    for b in LABELS:
        r = bucket_report[b]
        print(f"  {b:8s} precision={r['precision']:.3f}  recall={r['recall']:.3f}  f1={r['f1-score']:.3f}")

    print("\n" + "=" * 70)
    print(f"TRIAGE, cutoff chosen on validation for >={TARGET_RECALL:.0%} recall, applied to test")
    print("=" * 70)
    print(f"Cutoff (raw triage score): {cutoff:.4f}")
    print(f"Validation recall achieved at this cutoff: {achieved_recall_val:.4f}")
    print(f"Test precision at this cutoff:  {results['triage']['test_precision_at_cutoff']:.4f}")
    print(f"Test recall at this cutoff:     {results['triage']['test_recall_at_cutoff']:.4f}")
    print(f"Share of test reports flagged:  {results['triage']['test_flag_rate']:.1%}")
    print(f"AUPRC (ranking quality, cutoff-independent): {results['triage']['auprc']:.4f}")
    print(f"AUROC (comparable to the sponsor's number regardless of their threshold): {results['triage']['auroc']:.4f}")
    print(f"Brier score, raw probability:        {brier_raw:.4f}")
    print(f"Brier score, calibrated probability: {brier_calibrated:.4f}  (lower is better; calibration fixes the train/test harm-rate mismatch)")

    print(f"\nSaved model + evaluation to {OUT_DIR}/")

    # --- 6. Prove the saved artifacts work standalone ---
    print("\n" + "=" * 70)
    print("SANITY CHECK: reload from disk, predict on one new example report")
    print("=" * 70)
    demo_predict()


def demo_predict():
    import pandas as pd

    vectorizer = joblib.load(f"{OUT_DIR}/vectorizer.joblib")
    clf = joblib.load(f"{OUT_DIR}/model.joblib")
    calibrator = joblib.load(f"{OUT_DIR}/calibrator.joblib")
    cfg = json.load(open(f"{OUT_DIR}/config.json"))

    example = pd.DataFrame([{
        "event_comments": "Patient found on floor at 0300, alert and oriented, "
                           "complaining of left hip pain. X-ray confirmed fracture. "
                           "Transferred to OR for repair.",
        "Location Name": "ICU",
        "Encounter Service": "Cardiology",
        "Age at Encounter": 74,
        "ME - Prescribed - Name *... Name": "Warfarin",
        "ME - Prescribed - Dose *": "5mg",
        "ME - Admin - Name": None, "ME - Admin - Dose": None,
        "ADR - Suspect Med Name": None, "ADR - Dose *": None,
    }])
    text = build_input_text(example)
    X = vectorizer.transform(text)
    proba = clf.predict_proba(X)

    grade = FULL_LABELS[clf.predict(X)[0]]
    bucket_p = bucket_probs(proba, clf.classes_)[0]
    bucket = LABELS[bucket_p.argmax()]
    triage_raw = triage_score(proba, clf.classes_)[0]
    triage_calibrated = calibrator.predict_proba([[triage_raw]])[0, 1]
    flagged = triage_raw >= cfg["cutoff_raw_triage"]

    print(f"Input: {text.iloc[0][:80]}...")
    print(f"Predicted letter grade: {grade}")
    print(f"Derived 3-bucket:       {bucket}")
    print(f"Triage score (raw):     {triage_raw:.3f}")
    print(f"Triage score (calibrated probability of harm): {triage_calibrated:.3f}")
    print(f"Flagged for priority review: {flagged}")


if __name__ == "__main__":
    main()
