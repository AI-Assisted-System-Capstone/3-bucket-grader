"""
Error analysis: which true-harm reports (some/serious) does the DistilBERT
model currently misclassify as "none", and is there a pattern?

Reuses the exact same data prep and train/val/test split as
finetune_distilbert.py (same random_state=42) so this analyzes the same
held-out test set the reported metrics came from, then loads the trained
checkpoint to get per-example predictions and probabilities.
"""

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification

DATA_PATH = "/Users/devanshigupta/Downloads/Capstone Engineering Files/LOCAL_ONLY_student_facing_candidate.parquet"
HARM_COL = "Significance (PSRS Harm score)"
CHECKPOINT = "./distilbert_output/checkpoint-3016"

LABELS = ["none", "some", "serious"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for i, l in enumerate(LABELS)}

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

    # same split as finetune_distilbert.py / baseline_grader.py
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["bucket"]
    )

    print("Test set size:", len(test_df))

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
    model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT).to(device)
    model.eval()

    texts = test_df["text"].tolist()
    all_probs = []
    batch_size = 64
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tokenizer(
                batch, truncation=True, max_length=128, padding=True, return_tensors="pt"
            ).to(device)
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(probs)
    probs = np.concatenate(all_probs, axis=0)
    preds = probs.argmax(axis=1)

    test_df = test_df.reset_index(drop=True)
    test_df["pred"] = [ID2LABEL[p] for p in preds]
    test_df["prob_none"] = probs[:, LABEL2ID["none"]]
    test_df["prob_some"] = probs[:, LABEL2ID["some"]]
    test_df["prob_serious"] = probs[:, LABEL2ID["serious"]]
    test_df["word_count"] = test_df["text"].str.split().str.len()

    fn = test_df[(test_df["bucket"] != "none") & (test_df["pred"] == "none")].copy()
    print(f"\nFalse negatives (true harm predicted as 'none'): {len(fn)} / {len(test_df)} test rows")
    print(fn["bucket"].value_counts())

    print("\n--- Word count: false negatives vs. correctly-caught harm cases ---")
    caught = test_df[(test_df["bucket"] != "none") & (test_df["pred"] != "none")]
    print("FN median word count:", fn["word_count"].median(), " mean:", fn["word_count"].mean().round(1))
    print("Caught median word count:", caught["word_count"].median(), " mean:", caught["word_count"].mean().round(1))

    print("\n--- How close were the false negatives? (prob assigned to true bucket) ---")
    fn["prob_true_bucket"] = fn.apply(lambda r: r[f"prob_{r['bucket']}"], axis=1)
    print(fn["prob_true_bucket"].describe())
    near_misses = (fn["prob_true_bucket"] > 0.3).sum()
    print(f"Near-misses (model gave true bucket >0.3 prob but argmax still picked none): {near_misses}")

    print("\n--- Empty-field rates in false negatives vs. caught cases ---")
    for col in TEXT_COLS:
        fn_empty_rate = test_df.loc[fn.index, col].isna().mean()
        caught_empty_rate = test_df.loc[caught.index, col].isna().mean()
        print(f"  {col}: FN empty {fn_empty_rate:.1%}  |  caught empty {caught_empty_rate:.1%}")

    print("\n--- Sample false negatives (up to 15) ---")
    cols_to_show = ["bucket", "pred", "prob_none", "prob_some", "prob_serious", "word_count", "text"]
    with pd.option_context("display.max_colwidth", 200):
        for _, row in fn.sort_values("prob_true_bucket").head(15).iterrows():
            print(f"\n[true={row['bucket']} pred={row['pred']} "
                  f"p(none)={row['prob_none']:.2f} p(some)={row['prob_some']:.2f} p(serious)={row['prob_serious']:.2f} "
                  f"words={row['word_count']}]")
            print(row["text"][:400])

    fn.to_csv("./distilbert_output/false_negatives.csv", index=False)
    print("\nSaved all false negatives to ./distilbert_output/false_negatives.csv")


if __name__ == "__main__":
    main()
