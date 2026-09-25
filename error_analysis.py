"""
Error analysis: which true-harm reports (some/serious) does the DistilBERT
model currently misclassify as "none", and is there a pattern?

Reuses the same leakage-safe input / time-based split as finetune_distilbert.py
(data_pipeline.py) so this analyzes the same held-out Oct test set the
reported metrics came from, then loads the trained checkpoint to get
per-example predictions and probabilities.

NOTE: CHECKPOINT below must point at a checkpoint trained by the CURRENT
finetune_distilbert.py (leakage-safe, date-split). Checkpoints trained
before that fix used manager_comments/unit_actions_taken and a random split,
and are not comparable to the test set built here.
"""

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from data_pipeline import train_val_test, LABEL2ID, LABELS

CHECKPOINT = "./distilbert_output/checkpoint-3016"  # update after re-running finetune_distilbert.py

ID2LABEL = {i: l for i, l in enumerate(LABELS)}


def main():
    _, _, test_df = train_val_test()
    test_df = test_df.copy()

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
