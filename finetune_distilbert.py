"""
Step 2: a real context-aware model (DistilBERT) on the same 3-bucket task
and the SAME leakage-safe input / time-based split as baseline_grader.py,
for a fair comparison. See data_pipeline.py for details on both fixes:
manager_comments/unit_actions_taken are dropped (post-investigation leakage),
and the split is the dataset's real Jan-Aug/Sep/Oct split, not a random one --
matching the teammate's 11-buckets pipeline so results are comparable on the
same test rows.
"""

import json
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)

from data_pipeline import train_val_test, LABELS, LABEL2ID

OUT_DIR = "./distilbert_output"

MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 128  # a guess based on average narrative length -- NOT verified against
                   # the actual token-length distribution. See README "open questions".
ID2LABEL = {i: l for i, l in enumerate(LABELS)}


def main():
    train_df, val_df, test_df = train_val_test()

    print("Train:", train_df.shape, "Val:", val_df.shape, "Test:", test_df.shape)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("Using device:", device)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH, padding=False)

    train_ds = Dataset.from_pandas(train_df[["text", "label"]].reset_index(drop=True)).map(tokenize_fn, batched=True)
    val_ds = Dataset.from_pandas(val_df[["text", "label"]].reset_index(drop=True)).map(tokenize_fn, batched=True)
    test_ds = Dataset.from_pandas(test_df[["text", "label"]].reset_index(drop=True)).map(tokenize_fn, batched=True)

    train_ds = train_ds.rename_column("label", "labels")
    val_ds = val_ds.rename_column("label", "labels")
    test_ds = test_ds.rename_column("label", "labels")

    keep_cols = ["input_ids", "attention_mask", "labels"]
    train_ds.set_format(type="torch", columns=keep_cols)
    val_ds.set_format(type="torch", columns=keep_cols)
    test_ds.set_format(type="torch", columns=keep_cols)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID
    )

    # class weights, same idea as the baseline's class_weight="balanced"
    counts = train_df["label"].value_counts().reindex(range(len(LABELS)), fill_value=0).to_numpy(dtype=np.float32)
    inv_freq = 1.0 / np.where(counts == 0, 1e-6, counts)
    class_weights = inv_freq / inv_freq.sum() * len(LABELS)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
    print("Class weights:", dict(zip(LABELS, class_weights.tolist())))

    class WeightedTrainer(Trainer):
        def __init__(self, class_weights=None, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.class_weights = class_weights

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            weight = self.class_weights.to(logits.device) if self.class_weights is not None else None
            loss = torch.nn.functional.cross_entropy(logits, labels, weight=weight)
            return (loss, outputs) if return_outputs else loss

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        report = classification_report(labels, preds, target_names=LABELS, output_dict=True, zero_division=0)
        return {
            "accuracy": report["accuracy"],
            "serious_recall": report["serious"]["recall"],
            "serious_precision": report["serious"]["precision"],
            "macro_f1": report["macro avg"]["f1-score"],
        }

    training_args = TrainingArguments(
        output_dir=OUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_steps=100,
        load_best_model_at_end=True,
        # NOTE: optimizing checkpoint selection for serious-class recall was a
        # judgment call, not a team decision -- it may be trading away "some"
        # bucket recall (see README).
        metric_for_best_model="serious_recall",
        greater_is_better=True,
        save_total_limit=1,
        use_mps_device=(device == "mps"),
        report_to=[],
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        class_weights=class_weights_tensor,
    )

    trainer.train()

    test_results = trainer.predict(test_ds)
    logits = test_results.predictions
    labels = test_results.label_ids
    preds = np.argmax(logits, axis=1)

    print("\n=== DistilBERT on held-out test set ===")
    report = classification_report(labels, preds, target_names=LABELS, zero_division=0)
    print(report)
    cm = confusion_matrix(labels, preds, labels=list(range(len(LABELS))))
    print("Confusion matrix (rows=true, cols=pred), label order:", LABELS)
    print(cm)

    results = {
        "classification_report": classification_report(labels, preds, target_names=LABELS, output_dict=True, zero_division=0),
        "confusion_matrix": cm.tolist(),
        "label_order": LABELS,
    }
    with open(f"{OUT_DIR}/test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved results to", f"{OUT_DIR}/test_results.json")


if __name__ == "__main__":
    main()
