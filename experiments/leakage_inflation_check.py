"""
Quantifies how much the post-investigation fields (manager_comments,
unit_actions_taken) affect triage metrics, by training the SAME architecture
as severity_model.py twice: once leakage-safe (the production pipeline),
once with those two fields put back into the text input.

IMPORTANT SCOPE CAVEAT, added 2026-09-25: this does NOT tell us whether the
sponsor's specific reported numbers (95.2% sensitivity / 96.0% specificity)
are inflated. Their training script (LLM_fine_tuning_MIDA_PSRS_score_Weights_
clean.py) has not actually been shared in this project -- it was referenced
in a review, not attached. Until that script is available, "does leakage
inflate THEIR number" is unanswerable; this script only answers a narrower,
adjacent question: does adding these two fields change triage metrics on a
leakage-safe architecture we do control, on THIS synthetic dataset. It's
real evidence about the mechanism, not a stand-in for the real comparison.
It's also weaker evidence on synthetic data than it would be on real data:
manager comments on real reports narrate the actual resolution ("no further
action, patient counseled"), while this dataset's version may be more
generic templated text carrying less label signal -- which would itself
explain why the effect measured here is smaller than expected.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, precision_score, confusion_matrix, average_precision_score, roc_auc_score

from data_pipeline import DATA_PATH, ID_COL, HARM_COL, train_val_test, BUCKET_GROUPS, FULL_LABELS, HURT_GRADES

LEAKY_COLS = ["manager_comments", "unit_actions_taken"]


def build_leaky_text(raw_df, safe_text):
    """Same safe text, with the two leaky fields appended -- isolates the
    effect of adding leakage, rather than changing anything else."""
    mgr = raw_df["manager_comments"].fillna("").astype(str).str.strip()
    act = raw_df["unit_actions_taken"].fillna("").astype(str).str.strip()
    extra = ("Manager Comments: " + mgr).where(mgr != "", "") + " " + \
            ("Unit Actions Taken: " + act).where(act != "", "")
    return (safe_text + " " + extra).str.strip()


def evaluate_at_95_recall(y_hurt_val, triage_val, y_hurt_test, triage_test):
    order = np.argsort(triage_val)[::-1]
    cum_pos = np.cumsum(y_hurt_val[order])
    recall_at_k = cum_pos / y_hurt_val.sum()
    idx = min(np.searchsorted(recall_at_k, 0.95), len(triage_val) - 1)
    cutoff = triage_val[order][idx]

    flagged = (triage_test >= cutoff).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_hurt_test, flagged).ravel()
    specificity = tn / (tn + fp)
    return {
        "cutoff": float(cutoff),
        "recall": recall_score(y_hurt_test, flagged),
        "precision": precision_score(y_hurt_test, flagged),
        "specificity": specificity,
        "flag_rate": float(flagged.mean()),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        # Threshold-independent -- one noisy point isn't enough to judge this on.
        "auprc": average_precision_score(y_hurt_test, triage_test),
        "auroc": roc_auc_score(y_hurt_test, triage_test),
    }


def main():
    train_df, val_df, test_df = train_val_test()

    # Re-read raw parquet just for the two leaky columns, join back by row
    # position within each split (train_val_test preserves source row order
    # within each split via reset_index(drop=True) on a stable filter).
    raw = pd.read_parquet(DATA_PATH, columns=[ID_COL, HARM_COL, "manager_comments", "unit_actions_taken"])
    raw = raw.set_index(ID_COL)

    def attach_leaky_text(split_df):
        leaky_cols = raw.loc[split_df["event_no"], LEAKY_COLS].reset_index(drop=True)
        return build_leaky_text(leaky_cols, split_df["text"])

    train_leaky_text = attach_leaky_text(train_df)
    val_leaky_text = attach_leaky_text(val_df)
    test_leaky_text = attach_leaky_text(test_df)

    def fit_and_eval(train_text, val_text, test_text, label):
        # Same token_pattern fix as severity_model.py -- must match for a fair comparison.
        vectorizer = TfidfVectorizer(
            stop_words="english", max_features=20000, ngram_range=(1, 2), min_df=3,
            token_pattern=r"(?u)\b[\w-]+\b",
        )
        X_train = vectorizer.fit_transform(train_text)
        X_val = vectorizer.transform(val_text)
        X_test = vectorizer.transform(test_text)

        clf = LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=-1)
        clf.fit(X_train, train_df["full_label"])

        hurt_ids = set(BUCKET_GROUPS["some"]) | set(BUCKET_GROUPS["serious"])

        def triage(X):
            proba = clf.predict_proba(X)
            cols = [i for i, c in enumerate(clf.classes_) if c in hurt_ids]
            return proba[:, cols].sum(axis=1)

        triage_val = triage(X_val)
        triage_test = triage(X_test)
        result = evaluate_at_95_recall(
            val_df["hurt_label"].to_numpy(), triage_val,
            test_df["hurt_label"].to_numpy(), triage_test,
        )
        print(f"\n=== {label} ===")
        for k, v in result.items():
            print(f"  {k}: {v}")
        return result

    print("Training set size:", len(train_df), " Test set size:", len(test_df))
    print("True hurt cases in test:", int(test_df['hurt_label'].sum()), "/", len(test_df))

    safe = fit_and_eval(train_df["text"], val_df["text"], test_df["text"], "LEAKAGE-SAFE (production pipeline)")
    leaky = fit_and_eval(train_leaky_text, val_leaky_text, test_leaky_text, "WITH manager_comments + unit_actions_taken (leaky)")

    print("\n" + "=" * 70)
    print("EFFECT OF ADDING THE TWO POST-INVESTIGATION FIELDS")
    print("=" * 70)
    print(f"{'Metric':<15}{'Safe':>10}{'Leaky':>10}{'Delta':>10}")
    for k in ["recall", "precision", "specificity", "flag_rate", "auprc", "auroc"]:
        print(f"{k:<15}{safe[k]:>10.3f}{leaky[k]:>10.3f}{leaky[k]-safe[k]:>+10.3f}")
    print("\nScope: this measures the effect on THIS synthetic dataset and THIS")
    print("architecture only -- not a substitute for testing the sponsor's own")
    print("model without the leaky fields on real data.")


if __name__ == "__main__":
    main()
