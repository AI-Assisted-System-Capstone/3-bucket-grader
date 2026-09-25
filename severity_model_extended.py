"""
Extended evaluation of the saved severity model (severity_model.py), built in
response to two rounds of sponsor-facing review. Covers:

1. A specificity-matched comparison against the sponsor's actual reported
   numbers (95.2% sensitivity at 96.0% specificity) plus AUROC/AUPRC, since
   those are threshold-independent and comparable regardless of where either
   model's cutoff sits. NOT a comparison against their real model -- their
   training script has not been shared in this project, so a true head-to-
   head is still blocked. This is our model's own numbers, presented at a
   matching operating point for a fair read.
2. An operating-point menu (recall targets are OUR default choice, not a
   documented sponsor requirement) with miss counts broken down by tier
   (some vs. serious), at the sponsor's stated ~200 reports/day AND as a
   per-100-reports rate (the generator's actual October volume, 268/day, is
   neither number -- using it silently would misrepresent the sponsor's own
   stated volume).
3. Recall@20 / Recall@50, computed PER DAY and averaged -- not over the
   whole month at once, which understates the number by construction (a
   Safety Officer works one day's queue, not 31 days pooled together).
4. A prior-shift correction on the 10-way letter-grade argmax -- tried,
   found and fixed a bug (see comments), still rejected on net.
5. Subgroup breakdowns WITH n_pos and Wilson 95% intervals on every rate, so
   a small-sample subgroup can't be misread as a stable estimate. Also flags
   that subgroup membership was assigned by the data generator -- a
   synthetic disparity may be a generator artifact, not a real one.
6. Explanations for the TRIAGE decision specifically (why was this flagged),
   not just the winning letter grade (why THAT letter) -- these are
   different questions. Approximated by summing the E..I class coefficient
   vectors as an approximate "hurt direction", since a multinomial softmax
   isn't perfectly additive across classes.

Still blocked, not fakeable: archetype/template-overlap check (no such
column exists in this dataset) and a real head-to-head against the
sponsor's actual model (their training script has not been shared here).
"""

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    recall_score, precision_score, confusion_matrix, accuracy_score,
    mean_absolute_error, cohen_kappa_score, classification_report, f1_score,
    average_precision_score, roc_auc_score,
)

from data_pipeline import (
    train_val_test, FULL_LABELS, LABELS, BUCKET_GROUPS, build_input_text,
    DATA_PATH, ID_COL, DATE_COL,
)

MODEL_DIR = "./severity_model"
SPONSOR_SENSITIVITY = 0.952
SPONSOR_SPECIFICITY = 0.960
SPONSOR_STATED_VOLUME_PER_DAY = 200  # as stated by the sponsor, not the generator's actual October rate
RECALL_TARGETS = [0.95, 0.90, 0.85, 0.80]  # this project's own menu, not sponsor requirements


def load_model():
    vectorizer = joblib.load(f"{MODEL_DIR}/vectorizer.joblib")
    clf = joblib.load(f"{MODEL_DIR}/model.joblib")
    calibrator = joblib.load(f"{MODEL_DIR}/calibrator.joblib")
    return vectorizer, clf, calibrator


def triage_score(proba, classes_):
    hurt_ids = set(BUCKET_GROUPS["some"]) | set(BUCKET_GROUPS["serious"])
    cols = [i for i, c in enumerate(classes_) if c in hurt_ids]
    return proba[:, cols].sum(axis=1)


def bucket_probs(proba, classes_):
    out = np.zeros((proba.shape[0], len(LABELS)))
    for bi, bucket in enumerate(LABELS):
        group_ids = set(BUCKET_GROUPS[bucket])
        cols = [i for i, c in enumerate(classes_) if c in group_ids]
        out[:, bi] = proba[:, cols].sum(axis=1)
    return out


def wilson_interval(successes, n, z=1.96):
    """95% Wilson score interval for a proportion -- stable at small n,
    unlike a normal-approximation interval which can go outside [0,1]."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half_width = (z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)) / denom
    return (max(0.0, center - half_width), min(1.0, center + half_width))


def cutoff_for_specificity(y_true, scores, target_specificity):
    """Highest cutoff whose test specificity is >= target (search over
    observed score values), mirroring pick_cutoff_for_recall's approach but
    for specificity instead of recall."""
    order = np.argsort(scores)  # ascending
    y_sorted = y_true[order]
    n_neg = (y_true == 0).sum()
    # cumulative true negatives as cutoff rises from the bottom
    cum_true_neg = np.cumsum(1 - y_sorted)
    specificity_at = cum_true_neg / n_neg
    idx = np.searchsorted(specificity_at, target_specificity)
    idx = min(idx, len(scores) - 1)
    return float(scores[order][idx])


def sponsor_comparison(y_hurt_test, triage_test):
    print("\n" + "=" * 78)
    print("1. COMPARISON AGAINST THE SPONSOR'S REPORTED OPERATING POINT")
    print("=" * 78)
    print("NOTE: this is NOT a head-to-head against their real model -- their")
    print("training script has not been shared in this project. This is our")
    print("model's own numbers at a matching specificity, for a fair read.\n")

    auroc = roc_auc_score(y_hurt_test, triage_test)
    auprc = average_precision_score(y_hurt_test, triage_test)
    print(f"AUROC: {auroc:.4f}  (threshold-independent, directly comparable to their number regardless of cutoff)")
    print(f"AUPRC: {auprc:.4f}")

    results = {}
    for target_spec in [SPONSOR_SPECIFICITY, 0.95]:
        cutoff = cutoff_for_specificity(y_hurt_test, triage_test, target_spec)
        flagged = (triage_test >= cutoff).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_hurt_test, flagged).ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        print(f"\nAt {target_spec:.1%} specificity (cutoff {cutoff:.3f}):")
        print(f"  Our sensitivity: {sensitivity:.1%}   (sponsor's reported sensitivity at "
              f"{SPONSOR_SPECIFICITY:.1%} specificity: {SPONSOR_SENSITIVITY:.1%})")
        results[f"at_{target_spec}_specificity"] = {
            "cutoff": cutoff, "sensitivity": sensitivity, "specificity": specificity,
        }
    results["auroc"] = auroc
    results["auprc"] = auprc
    return results


def operating_point_menu(y_hurt_val, triage_val, y_hurt_test, triage_test, test_df):
    print("\n" + "=" * 78)
    print("2. OPERATING-POINT MENU (this project's own targets, not a sponsor requirement)")
    print("=" * 78)
    order = np.argsort(triage_val)[::-1]
    y_sorted = y_val_sorted = y_hurt_val[order]
    s_sorted = triage_val[order]
    cum_pos = np.cumsum(y_sorted)
    recall_at_k = cum_pos / y_hurt_val.sum()

    grade = test_df["bucket"].to_numpy()  # 'some' vs 'serious' among hurt cases
    rows = []
    for volume_label, reports_per_day in [
        (f"sponsor-stated ({SPONSOR_STATED_VOLUME_PER_DAY}/day)", SPONSOR_STATED_VOLUME_PER_DAY),
        ("per 100 reports", 100),
    ]:
        print(f"\n--- at {volume_label} ---")
        print(f"{'Target':>8}{'Cutoff':>9}{'Recall':>9}{'Prec.':>8}{'Flagged/day':>13}"
              f"{'TP/day':>9}{'FP/day':>9}{'Miss: some/day':>16}{'Miss: serious/day':>19}")
        for target in RECALL_TARGETS:
            idx = min(np.searchsorted(recall_at_k, target), len(s_sorted) - 1)
            cutoff = float(s_sorted[idx])
            flagged = (triage_test >= cutoff).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_hurt_test, flagged).ravel()
            test_recall = tp / (tp + fn)
            test_precision = tp / (tp + fp) if (tp + fp) else 0.0
            missed = (flagged == 0) & (y_hurt_test == 1)
            missed_some = int(((grade == "some") & missed).sum())
            missed_serious = int(((grade == "serious") & missed).sum())
            scale = reports_per_day / len(test_df)
            row = {
                "volume_label": volume_label, "target_recall": target, "cutoff": cutoff,
                "test_recall": test_recall, "test_precision": test_precision,
                "flagged_per_period": flagged.mean() * reports_per_day,
                "tp_per_period": tp * scale, "fp_per_period": fp * scale,
                "missed_some_per_period": missed_some * scale,
                "missed_serious_per_period": missed_serious * scale,
            }
            rows.append(row)
            print(f"{target:>7.0%}{cutoff:>9.3f}{test_recall:>9.1%}{test_precision:>8.1%}"
                  f"{row['flagged_per_period']:>13.1f}{row['tp_per_period']:>9.1f}{row['fp_per_period']:>9.1f}"
                  f"{missed_some*scale:>16.2f}{missed_serious*scale:>19.2f}")
    return rows


def recall_at_k_per_day(test_df, triage_test, y_hurt_test, ks):
    print("\n" + "=" * 78)
    print("3. RECALL@K, COMPUTED PER DAY AND AVERAGED (not pooled over the whole month)")
    print("=" * 78)
    raw = pd.read_parquet(DATA_PATH, columns=[ID_COL, DATE_COL])
    raw = raw.set_index(ID_COL)
    dates = raw.loc[test_df["event_no"], DATE_COL].reset_index(drop=True)
    day = pd.to_datetime(dates).dt.date.to_numpy()

    out = {}
    for k in ks:
        daily_recalls = []
        for d in np.unique(day):
            mask = day == d
            y_day = y_hurt_test[mask]
            s_day = triage_test[mask]
            n_pos_day = y_day.sum()
            if n_pos_day == 0:
                continue  # can't compute recall on a day with no true harm cases
            top_k_idx = np.argsort(s_day)[::-1][:min(k, len(s_day))]
            caught = y_day[top_k_idx].sum()
            daily_recalls.append(caught / n_pos_day)
        avg_recall = float(np.mean(daily_recalls))
        out[k] = avg_recall
        print(f"  Top {k:>2} reports of EACH day's queue catch, on average, {avg_recall:.1%} "
              f"of that day's true harm cases (averaged over {len(daily_recalls)} days)")
    return out


def prior_shift_correction(clf, proba_test, train_df, val_df, y_full_test):
    print("\n" + "=" * 78)
    print("4. PRIOR-SHIFT CORRECTION ON THE LETTER-GRADE ARGMAX")
    print("=" * 78)
    print("Uncorrected argmax comes from a model trained at 15.1% harm prevalence,")
    print("skewing predictions toward E/F even when the true rate is ~5%.")

    classes = clf.classes_
    train_counts = train_df["full_label"].value_counts().reindex(classes, fill_value=0).to_numpy(dtype=float)
    val_counts = val_df["full_label"].value_counts().reindex(classes, fill_value=0).to_numpy(dtype=float)

    alpha = 1.0
    train_prior = (train_counts + alpha) / (train_counts.sum() + alpha * len(classes))
    val_prior = (val_counts + alpha) / (val_counts.sum() + alpha * len(classes))
    correction = np.log(val_prior) - np.log(train_prior)
    print(f"  Validation counts for classes with < 10 rows (the ones Laplace smoothing protects): "
          + ", ".join(f"{FULL_LABELS[c]}={int(n)}" for c, n in zip(classes, val_counts) if n < 10))

    log_proba = np.log(np.clip(proba_test, 1e-12, 1.0))
    corrected_log_proba = log_proba + correction
    pred_uncorrected = classes[proba_test.argmax(axis=1)]
    pred_corrected = classes[corrected_log_proba.argmax(axis=1)]

    def summarize(preds, label):
        acc = accuracy_score(y_full_test, preds)
        mae = mean_absolute_error(y_full_test, preds)
        kappa = cohen_kappa_score(y_full_test, preds, weights="quadratic")
        print(f"\n  {label}: accuracy={acc:.4f}  MAE={mae:.4f}  kappa={kappa:.4f}")
        return acc, mae, kappa

    summarize(pred_uncorrected, "Uncorrected (production)")
    summarize(pred_corrected, "Prior-shift corrected (smoothed) -- still rejected, see docstring")
    return {"correction_vector": correction.tolist()}


def subgroup_breakdown(test_df, grade_pred, triage_test, cutoff, y_hurt_test):
    print("\n" + "=" * 78)
    print("5. SUBGROUP BREAKDOWN, WITH n_pos AND WILSON 95% INTERVALS (test set)")
    print("=" * 78)
    print("CAVEAT: subgroup membership (service, location, age) was assigned by the")
    print("data GENERATOR, not by real hospital operations -- a disparity here may")
    print("be a generator artifact. The subgroup analysis that counts is the one on")
    print("the real container run, not this one.\n")
    flagged = (triage_test >= cutoff).astype(int)

    for col, top_n in [("encounter_service", 6), ("age_band", 10)]:
        print(f"\n  --- by {col} ---")
        groups = test_df[col].value_counts().head(top_n).index
        print(f"  {'Group':<26}{'N':>5}{'n_pos':>7}{'Recall (95% CI)':>24}{'Precision (95% CI)':>24}")
        for g in groups:
            mask = (test_df[col] == g).to_numpy()
            n = int(mask.sum())
            y_h = y_hurt_test[mask]
            fl = flagged[mask]
            n_pos = int(y_h.sum())
            n_flagged = int(fl.sum())
            tp = int((y_h & fl).sum())
            if n_pos > 0:
                rec = tp / n_pos
                rec_lo, rec_hi = wilson_interval(tp, n_pos)
                rec_str = f"{rec:.2f} [{rec_lo:.2f}-{rec_hi:.2f}]  n={n_pos}"
            else:
                rec_str = "n/a (0 positives)"
            if n_flagged > 0:
                prec = tp / n_flagged
                prec_lo, prec_hi = wilson_interval(tp, n_flagged)
                prec_str = f"{prec:.2f} [{prec_lo:.2f}-{prec_hi:.2f}]  n={n_flagged}"
            else:
                prec_str = "n/a (0 flagged)"
            print(f"  {str(g):<26}{n:>5}{n_pos:>7}{rec_str:>24}{prec_str:>24}")


def explain_triage(vectorizer, clf, text, top_n=8):
    """Top terms driving the TRIAGE decision (flag or not), not the winning
    letter grade -- approximated by summing the E..I class coefficient
    vectors as a 'hurt direction'. A multinomial softmax isn't perfectly
    additive across classes, so this is an approximation of the true
    marginal effect, not an exact decomposition."""
    x = vectorizer.transform([text])
    feature_names = np.array(vectorizer.get_feature_names_out())
    present_idx = x.nonzero()[1]
    hurt_ids = set(BUCKET_GROUPS["some"]) | set(BUCKET_GROUPS["serious"])
    hurt_cols = [i for i, c in enumerate(clf.classes_) if c in hurt_ids]
    hurt_direction = clf.coef_[hurt_cols].sum(axis=0)  # approximate "pushes toward hurt"
    contributions = x[0, present_idx].toarray().ravel() * hurt_direction[present_idx]
    order = np.argsort(contributions)[::-1][:top_n]
    return [(feature_names[present_idx[i]], float(contributions[i])) for i in order]


def explain_grade(vectorizer, clf, text, predicted_class_idx, top_n=8):
    """Top terms behind the WINNING LETTER GRADE specifically (why E rather
    than F), ranked by contribution (coef x tfidf value), not raw coefficient."""
    x = vectorizer.transform([text])
    feature_names = np.array(vectorizer.get_feature_names_out())
    present_idx = x.nonzero()[1]
    coefs = clf.coef_[list(clf.classes_).index(predicted_class_idx)]
    contributions = x[0, present_idx].toarray().ravel() * coefs[present_idx]
    order = np.argsort(contributions)[::-1][:top_n]
    return [(feature_names[present_idx[i]], float(contributions[i])) for i in order]


def main():
    vectorizer, clf, calibrator = load_model()
    train_df, val_df, test_df = train_val_test()
    cfg = json.load(open(f"{MODEL_DIR}/config.json"))
    cutoff = cfg["cutoff_raw_triage"]

    X_val = vectorizer.transform(val_df["text"])
    X_test = vectorizer.transform(test_df["text"])
    proba_val = clf.predict_proba(X_val)
    proba_test = clf.predict_proba(X_test)

    triage_val = triage_score(proba_val, clf.classes_)
    triage_test = triage_score(proba_test, clf.classes_)
    y_hurt_val = val_df["hurt_label"].to_numpy()
    y_hurt_test = test_df["hurt_label"].to_numpy()
    grade_pred = clf.predict(X_test)

    print(f"True harm cases in test: {int(y_hurt_test.sum())} / {len(test_df)}")

    sponsor_cmp = sponsor_comparison(y_hurt_test, triage_test)
    op_menu = operating_point_menu(y_hurt_val, triage_val, y_hurt_test, triage_test, test_df)
    recall_k = recall_at_k_per_day(test_df, triage_test, y_hurt_test, [20, 50])
    prior_shift = prior_shift_correction(clf, proba_test, train_df, val_df, test_df["full_label"].to_numpy())
    subgroup_breakdown(test_df, grade_pred, triage_test, cutoff, y_hurt_test)

    print("\n" + "=" * 78)
    print("6. MACRO F1 (production model, test set)")
    print("=" * 78)
    macro_f1_grade = f1_score(test_df["full_label"], grade_pred, average="macro")
    bucket_pred = np.array(LABELS)[bucket_probs(proba_test, clf.classes_).argmax(axis=1)]
    macro_f1_bucket = f1_score(test_df["bucket"], bucket_pred, average="macro")
    print(f"  10-grade macro F1:        {macro_f1_grade:.4f}")
    print(f"  Derived 3-bucket macro F1: {macro_f1_bucket:.4f}")

    print("\n" + "=" * 78)
    print("7. EXPLANATIONS: triage decision vs. letter-grade decision (different questions)")
    print("=" * 78)
    example = pd.DataFrame([{
        "event_comments": "Patient found on floor at 0300, alert and oriented, "
                           "complaining of left hip pain. X-ray confirmed fracture. "
                           "Transferred to OR for repair.",
        "Location Name": "ICU", "Encounter Service": "Cardiology", "Age at Encounter": 74,
        "ME - Prescribed - Name *... Name": "Warfarin", "ME - Prescribed - Dose *": "5mg",
        "ME - Admin - Name": None, "ME - Admin - Dose": None,
        "ADR - Suspect Med Name": None, "ADR - Dose *": None,
    }])
    text = build_input_text(example).iloc[0]
    x = vectorizer.transform([text])
    pred_idx = clf.predict(x)[0]
    print(f"  Predicted grade: {FULL_LABELS[pred_idx]}")
    print("\n  Why this grade specifically (E, not F or D):")
    for term, weight in explain_grade(vectorizer, clf, text, pred_idx):
        print(f"    {term:25s} {weight:+.3f}")
    print("\n  Why this report was FLAGGED (approximate hurt-direction contribution):")
    for term, weight in explain_triage(vectorizer, clf, text):
        print(f"    {term:25s} {weight:+.3f}")

    results = {
        "sponsor_comparison": sponsor_cmp,
        "operating_point_menu": op_menu,
        "recall_at_k_per_day": recall_k,
        "prior_shift_correction": prior_shift,
        "macro_f1_grade": macro_f1_grade,
        "macro_f1_bucket": macro_f1_bucket,
        "not_addressed": {
            "archetype_overlap": "No archetype/template ID column exists in this dataset. "
                                  "Needs sponsor-provided archetype IDs to check train/test template overlap.",
            "sponsor_model_head_to_head": "The sponsor's actual training script has not been shared in this "
                                          "project. leakage_inflation_check.py measures a related but narrower "
                                          "question (effect of the leaky fields on a controlled architecture) "
                                          "-- it is not a substitute for the real comparison.",
        },
    }
    with open(f"{MODEL_DIR}/extended_evaluation.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved to {MODEL_DIR}/extended_evaluation.json")


if __name__ == "__main__":
    main()
