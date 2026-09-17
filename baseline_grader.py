"""
Step 1 baseline: can a dumb, fast word-counting model beat "always guess
the most common answer" at grading safety report severity?

Collapses the 9 harm grades into 3 simple buckets:
  none    = A, B1, B2, C, D   (unsafe condition / near miss / no harm)
  some    = E, F              (harm, treated)
  serious = G, H, I           (permanent harm / near death / death)

Trains TF-IDF + Logistic Regression on the free-text narrative fields.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.dummy import DummyClassifier

DATA_PATH = "/Users/devanshigupta/Downloads/Capstone Engineering Files/LOCAL_ONLY_student_facing_candidate.parquet"
HARM_COL = "Significance (PSRS Harm score)"

BUCKET_MAP = {
    "A-Unsafe Condition": "none",
    "B1-Near Miss (by chance)": "none",
    "B2-Near Miss (avoided)": "none",
    "C-No Harm (no monitoring)": "none",
    "D-No Harm (intervened)": "none",
    "E-Harm-Temp (treated/intervened)": "some",
    "F-Harm-Temp (add hospitalization)": "some",
    "G-Harm-Permanent": "serious",
    "H-Harm-Near Death": "serious",
    "I-Harm-Death": "serious",
}

TEXT_COLS = ["event_comments", "manager_comments", "unit_actions_taken"]
TEXT_COL_LABELS = {
    "event_comments": "Event Comments",
    "manager_comments": "Manager Comments",
    "unit_actions_taken": "Unit Actions Taken",
}


def build_labeled_text(df):
    labeled_parts = []
    for col in TEXT_COLS:
        series = df[col].fillna("").astype(str).str.strip()
        prefix = TEXT_COL_LABELS[col] + ": "
        labeled_parts.append((prefix + series).where(series != "", ""))
    return pd.concat(labeled_parts, axis=1).agg(" ".join, axis=1).str.replace(
        r"\s+", " ", regex=True
    ).str.strip()


def main():
    df = pd.read_parquet(DATA_PATH)

    df = df[df[HARM_COL].notna()].copy()
    df["bucket"] = df[HARM_COL].map(BUCKET_MAP)
    df = df[df["bucket"].notna()].copy()

    df["text"] = build_labeled_text(df)
    df = df[df["text"].str.len() > 0].copy()

    print("Rows used:", len(df))
    print("\nBucket distribution:")
    print(df["bucket"].value_counts())
    print(df["bucket"].value_counts(normalize=True).round(3))

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["bucket"], test_size=0.2, random_state=42, stratify=df["bucket"]
    )

    # --- Baseline: always guess the most common bucket ---
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    dummy_preds = dummy.predict(X_test)
    print("\n=== Dummy baseline (always guess most common) ===")
    print(classification_report(y_test, dummy_preds, zero_division=0))

    # --- Simple model: word counts + logistic regression ---
    vectorizer = TfidfVectorizer(
        stop_words="english", max_features=20000, ngram_range=(1, 2), min_df=3
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    clf = LogisticRegression(
        max_iter=1000, class_weight="balanced", n_jobs=-1
    )
    clf.fit(X_train_vec, y_train)
    preds = clf.predict(X_test_vec)

    print("\n=== TF-IDF + Logistic Regression ===")
    print(classification_report(y_test, preds, zero_division=0))
    print("Confusion matrix (rows=true, cols=pred), labels order:", sorted(y_test.unique()))
    print(confusion_matrix(y_test, preds, labels=sorted(y_test.unique())))

    # --- Which words matter most for "serious" ---
    serious_idx = list(clf.classes_).index("serious")
    coefs = clf.coef_[serious_idx]
    feature_names = vectorizer.get_feature_names_out()
    top_pos = coefs.argsort()[-20:][::-1]
    print("\nTop words/phrases pushing toward 'serious':")
    for i in top_pos:
        print(f"  {feature_names[i]:30s} {coefs[i]:.3f}")


if __name__ == "__main__":
    main()
