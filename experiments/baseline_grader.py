"""
Step 1 baseline: can a dumb, fast word-counting model beat "always guess
the most common answer" at grading safety report severity?

Collapses the 9 harm grades into 3 simple buckets:
  none    = A, B1, B2, C, D   (unsafe condition / near miss / no harm)
  some    = E, F              (harm, treated)
  serious = G, H, I           (permanent harm / near death / death)

Trains TF-IDF + Logistic Regression on `event_comments` plus a short intake-
field prefix (unit, service, age, medications) -- NOT manager_comments or
unit_actions_taken, which are filled in after a safety officer investigates
and would leak the answer. Split is the dataset's real time-based split
(Jan-Aug train / Sep validation / Oct test), not a random one, so results
are comparable to the teammate's 11-buckets event-type/hurt model on the
same rows. See data_pipeline.py for details.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.dummy import DummyClassifier

from data_pipeline import train_val_test, LABELS


def main():
    train_df, val_df, test_df = train_val_test()
    print("Train:", train_df.shape, "Val:", val_df.shape, "Test:", test_df.shape)
    print("\nTrain bucket distribution:")
    print(train_df["bucket"].value_counts())
    print(train_df["bucket"].value_counts(normalize=True).round(3))

    X_train, y_train = train_df["text"], train_df["bucket"]
    X_val, y_val = val_df["text"], val_df["bucket"]
    X_test, y_test = test_df["text"], test_df["bucket"]

    # --- Baseline: always guess the most common bucket ---
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    print("\n=== Dummy baseline (always guess most common), test set ===")
    print(classification_report(y_test, dummy.predict(X_test), zero_division=0))

    # --- Simple model: word counts + logistic regression ---
    vectorizer = TfidfVectorizer(
        stop_words="english", max_features=20000, ngram_range=(1, 2), min_df=3
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)
    X_test_vec = vectorizer.transform(X_test)

    clf = LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=-1)
    clf.fit(X_train_vec, y_train)

    print("\n=== TF-IDF + Logistic Regression, validation set (Sep) ===")
    print(classification_report(y_val, clf.predict(X_val_vec), labels=LABELS, zero_division=0))

    test_preds = clf.predict(X_test_vec)
    print("\n=== TF-IDF + Logistic Regression, test set (Oct) ===")
    print(classification_report(y_test, test_preds, labels=LABELS, zero_division=0))
    print("Confusion matrix (rows=true, cols=pred), labels order:", LABELS)
    print(confusion_matrix(y_test, test_preds, labels=LABELS))

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
