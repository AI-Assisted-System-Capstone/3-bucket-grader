"""
The real experiment behind the "3 buckets vs. 11 buckets" question, once you
strip away the false choice: does a model trained on the FULL ordinal PSRS
scale (A, B1, B2, C, D, E, F, G, H, I) give a BETTER or WORSE triage ranking
than a model trained DIRECTLY on the coarse binary target (hurt = E-I)?

This matters because the sponsor needs both a triage score (recall@k / AUPRC
on the E-I line, for worklist ranking) and a pre-filled PSRS letter (ordinal
metrics: MAE, quadratic-weighted kappa, exact match), from slides 4-5/34.
Training on the fine-grained scale and marginalizing (summing softmax over
E..I, same trick as the sponsor's own `p_ge6 = probs[:, 6:].sum(axis=1)`)
gets you both outputs from one model -- IF the fine-grained model's summed
probabilities rank as well as a model built to rank hurt/not-hurt directly.
They might not: G/H/I have only 84/144/167 rows in the full 80k dataset, so
an 11-way softmax could be poorly calibrated there in a way that hurts the
marginalized triage score, even if the fine-grained letter predictions
themselves are fine on average.

Three things measured on the same Oct test set:

1. Triage (E-I) ranking: AUPRC of
     (a) fine-grained model's marginalized P(E..I) vs.
     (b) a model trained directly on the binary hurt_label
   Also recall among the top 10% highest-risk-ranked reports (a "worklist"
   framing), for both.

2. Pre-fill quality (fine-grained model only, there's no coarse equivalent):
   exact-match accuracy, MAE (treating A..I as ordinal ints 0-9), and
   quadratic-weighted kappa, all vs. the majority-class baseline.

3. Rare-class calibration: precision/recall for G, H, I specifically from the
   fine-grained model, since that's exactly where the sponsor's calibration
   concern applies.

Model: TF-IDF (same vectorizer settings as baseline_grader.py) + Logistic
Regression, multinomial for the fine-grained model, binary for the direct
model. Same leakage-safe / date-split data as everything else here.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    cohen_kappa_score,
    accuracy_score,
    classification_report,
)

from data_pipeline import train_val_test, FULL_LABELS, FULL_LABEL2ID


def recall_at_top_k_pct(y_true, scores, pct=0.10):
    n = len(scores)
    k = max(1, int(round(n * pct)))
    top_idx = np.argsort(scores)[::-1][:k]
    return y_true[top_idx].sum() / y_true.sum(), k


def main():
    train_df, val_df, test_df = train_val_test()
    print("Train:", train_df.shape, "Val:", val_df.shape, "Test:", test_df.shape)
    print("\nFull-grade distribution (train):")
    print(train_df["grade"].value_counts())

    vectorizer = TfidfVectorizer(stop_words="english", max_features=20000, ngram_range=(1, 2), min_df=3)
    X_train = vectorizer.fit_transform(train_df["text"])
    X_test = vectorizer.transform(test_df["text"])

    y_full_train = train_df["full_label"].to_numpy()
    y_full_test = test_df["full_label"].to_numpy()
    y_hurt_train = train_df["hurt_label"].to_numpy()
    y_hurt_test = test_df["hurt_label"].to_numpy()

    # --- Model A: fine-grained, 10-class ordinal (A..I) ---
    clf_full = LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=-1)
    clf_full.fit(X_train, y_full_train)
    proba_full = clf_full.predict_proba(X_test)  # (n, n_classes_present)
    pred_full = clf_full.predict(X_test)

    # class_weight/predict_proba columns follow clf_full.classes_, which may
    # not include every one of the 10 labels if a rare grade never appears in
    # train (shouldn't happen here, but don't assume the identity mapping).
    hurt_ids = {FULL_LABEL2ID[g] for g in FULL_LABELS[5:]}  # E,F,G,H,I are indices 5-9
    hurt_col_idx = [i for i, c in enumerate(clf_full.classes_) if c in hurt_ids]
    p_hurt_marginal = proba_full[:, hurt_col_idx].sum(axis=1)

    # --- Model B: direct binary hurt/not-hurt ---
    clf_hurt = LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=-1)
    clf_hurt.fit(X_train, y_hurt_train)
    p_hurt_direct = clf_hurt.predict_proba(X_test)[:, list(clf_hurt.classes_).index(1)]

    # --- 1. Triage ranking: AUPRC + recall@top-10% ---
    auprc_marginal = average_precision_score(y_hurt_test, p_hurt_marginal)
    auprc_direct = average_precision_score(y_hurt_test, p_hurt_direct)
    recall_marginal, k = recall_at_top_k_pct(y_hurt_test, p_hurt_marginal)
    recall_direct, _ = recall_at_top_k_pct(y_hurt_test, p_hurt_direct)
    base_rate = y_hurt_test.mean()

    print("\n" + "=" * 70)
    print("1. TRIAGE RANKING (E-I vs. A-D), test set")
    print("=" * 70)
    print(f"Base rate (hurt): {base_rate:.4f}  ({int(y_hurt_test.sum())}/{len(y_hurt_test)})")
    print(f"AUPRC, marginalized from fine-grained model: {auprc_marginal:.4f}")
    print(f"AUPRC, direct binary model:                  {auprc_direct:.4f}")
    print(f"Recall@top-{k} ({k/len(y_hurt_test):.0%} of test), marginalized: {recall_marginal:.4f}")
    print(f"Recall@top-{k} ({k/len(y_hurt_test):.0%} of test), direct:       {recall_direct:.4f}")

    # --- 2. Pre-fill quality (fine-grained model vs. majority baseline) ---
    dummy = DummyClassifier(strategy="most_frequent").fit(X_train, y_full_train)
    pred_dummy = dummy.predict(X_test)

    acc_full = accuracy_score(y_full_test, pred_full)
    acc_dummy = accuracy_score(y_full_test, pred_dummy)
    mae_full = mean_absolute_error(y_full_test, pred_full)  # ordinal ints, since FULL_LABEL2ID order is severity order
    mae_dummy = mean_absolute_error(y_full_test, pred_dummy)
    kappa_full = cohen_kappa_score(y_full_test, pred_full, weights="quadratic")
    kappa_dummy = cohen_kappa_score(y_full_test, pred_dummy, weights="quadratic")

    print("\n" + "=" * 70)
    print("2. PRE-FILL QUALITY (exact PSRS letter, ordinal A..I), test set")
    print("=" * 70)
    print(f"Exact-match accuracy: model={acc_full:.4f}  majority-baseline={acc_dummy:.4f}")
    print(f"MAE (ordinal grade):  model={mae_full:.4f}  majority-baseline={mae_dummy:.4f}  (lower is better)")
    print(f"Quadratic-weighted kappa: model={kappa_full:.4f}  majority-baseline={kappa_dummy:.4f}  (higher is better, 0=chance)")

    # --- 3. Rare-class calibration: G, H, I specifically ---
    print("\n" + "=" * 70)
    print("3. RARE-CLASS CALIBRATION (fine-grained model), test set")
    print("=" * 70)
    present_labels = sorted(set(y_full_test) | set(pred_full))
    target_names = [FULL_LABELS[i] for i in present_labels]
    report = classification_report(y_full_test, pred_full, labels=present_labels,
                                    target_names=target_names, zero_division=0)
    print(report)
    for grade in ["G-Harm-Permanent", "H-Harm-Near Death", "I-Harm-Death"]:
        gid = FULL_LABEL2ID[grade]
        n_test = int((y_full_test == gid).sum())
        n_train = int((y_full_train == gid).sum())
        print(f"  {grade}: {n_train} train rows, {n_test} test rows")


if __name__ == "__main__":
    main()
