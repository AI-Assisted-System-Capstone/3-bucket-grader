"""
Step 2: a real context-aware model (DistilBERT) on the SAME leakage-safe
input / time-based split as severity_model.py, for a fair comparison. See
data_pipeline.py for details on both fixes: manager_comments/
unit_actions_taken are dropped (post-investigation leakage), and the split is
the dataset's real Jan-Aug/Sep/Oct split, not a random one -- matching the
teammate's 11-buckets pipeline so results are comparable on the same test rows.

Two knobs, both defaulting to the original run (3-bucket, 128 tokens):

  --target bucket   train directly on none/some/serious (the original run)
  --target grade    train on all 10 PSRS grades (A..I), then derive bucket and
                    triage outputs exactly the way production does
                    (severity_model.bucket_probs / triage_score)
  --max-length N    tokens kept per report. 128 truncates ~25% of reports;
                    p95 is ~228 and 256 covers ~98% (see PIPELINE.md)

Either way the evaluation matches severity_model.py's: triage score =
P(some) + P(serious), cutoff picked on September for >=95% recall and frozen,
Platt calibration fit on September, everything reported on October, plus the
sponsor's 96%-specificity comparison from severity_model_extended.py. A
summary is written to distilbert_results/<target>_len<N>.json (committed);
checkpoints go to distilbert_output/ (gitignored).
"""

import argparse
import json
import os

import numpy as np
import torch
from datasets import Dataset
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    mean_absolute_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from data_pipeline import BUCKET_MAP, FULL_LABELS, LABEL2ID, LABELS, train_val_test
from severity_model import TARGET_RECALL, bucket_probs, pick_cutoff_for_recall, triage_score
from severity_model_extended import SPONSOR_SPECIFICITY, cutoff_for_specificity

OUT_DIR = "./distilbert_output"
RESULTS_DIR = "./distilbert_results"

MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 128  # the original guess; pass --max-length to change it

TARGETS = {
    # target: (label column, class names)
    "bucket": ("label", LABELS),
    "grade": ("full_label", FULL_LABELS),
}

# grade id -> bucket id, for scoring a 10-grade model's labels as buckets
GRADE_TO_BUCKET = np.array([LABEL2ID[BUCKET_MAP[g]] for g in FULL_LABELS])


def to_bucket_probs(proba, target):
    """(n, n_classes) probabilities -> (n, 3) bucket probabilities."""
    if target == "bucket":
        return proba
    return bucket_probs(proba, np.arange(len(FULL_LABELS)))


def to_triage(proba, target):
    if target == "bucket":
        return proba[:, 1] + proba[:, 2]
    return triage_score(proba, np.arange(len(FULL_LABELS)))


def evaluate(target, proba_val, proba_test, val_df, test_df):
    """Same derived outputs and metrics as severity_model.py's evaluation.json."""
    y_hurt_val = val_df["hurt_label"].to_numpy()
    y_hurt_test = test_df["hurt_label"].to_numpy()
    y_bucket_test = test_df["label"].to_numpy()

    bucket_pred_test = to_bucket_probs(proba_test, target).argmax(axis=1)
    triage_val = to_triage(proba_val, target)
    triage_test = to_triage(proba_test, target)

    cutoff = pick_cutoff_for_recall(y_hurt_val, triage_val, TARGET_RECALL)
    flagged_test = (triage_test >= cutoff).astype(int)

    calibrator = LogisticRegression()
    calibrator.fit(triage_val.reshape(-1, 1), y_hurt_val)
    calibrated_test = calibrator.predict_proba(triage_test.reshape(-1, 1))[:, 1]

    spec_cutoff = cutoff_for_specificity(y_hurt_test, triage_test, SPONSOR_SPECIFICITY)
    spec_flag = triage_test >= spec_cutoff
    tn, fp, fn, tp = confusion_matrix(y_hurt_test, spec_flag).ravel()

    results = {
        "bucket_3way_derived": {
            "per_bucket": classification_report(
                y_bucket_test, bucket_pred_test, labels=list(range(len(LABELS))),
                target_names=LABELS, output_dict=True, zero_division=0,
            ),
            "confusion_matrix": confusion_matrix(
                y_bucket_test, bucket_pred_test, labels=list(range(len(LABELS)))).tolist(),
        },
        "triage": {
            "auprc": average_precision_score(y_hurt_test, triage_test),
            "auroc": roc_auc_score(y_hurt_test, triage_test),
            "cutoff_chosen_on_validation": cutoff,
            "recall_on_validation_at_cutoff": recall_score(y_hurt_val, (triage_val >= cutoff).astype(int)),
            "test_precision_at_cutoff": precision_score(y_hurt_test, flagged_test),
            "test_recall_at_cutoff": recall_score(y_hurt_test, flagged_test),
            "test_flag_rate": float(flagged_test.mean()),
            "serious_missed_at_cutoff": int(((test_df["bucket"] == "serious").to_numpy() & (flagged_test == 0)).sum()),
            "brier_score_raw": brier_score_loss(y_hurt_test, triage_test),
            "brier_score_calibrated": brier_score_loss(y_hurt_test, calibrated_test),
            f"sensitivity_at_{SPONSOR_SPECIFICITY}_specificity": tp / (tp + fn),
            f"specificity_at_{SPONSOR_SPECIFICITY}_specificity": tn / (tn + fp),
        },
    }
    if target == "grade":
        y_full_test = test_df["full_label"].to_numpy()
        grade_pred_test = proba_test.argmax(axis=1)
        present = sorted(set(y_full_test) | set(grade_pred_test))
        results["grade_10way"] = {
            "exact_match_accuracy": accuracy_score(y_full_test, grade_pred_test),
            "mae_ordinal": mean_absolute_error(y_full_test, grade_pred_test),
            "quadratic_weighted_kappa": cohen_kappa_score(y_full_test, grade_pred_test, weights="quadratic"),
            "macro_f1": classification_report(
                y_full_test, grade_pred_test, labels=present, output_dict=True, zero_division=0,
            )["macro avg"]["f1-score"],
        }
    return results


def print_summary(results):
    if "grade_10way" in results:
        g = results["grade_10way"]
        print(f"10-grade: acc={g['exact_match_accuracy']:.4f}  MAE={g['mae_ordinal']:.4f}  "
              f"QWK={g['quadratic_weighted_kappa']:.4f}  macro F1={g['macro_f1']:.4f}")
    rep = results["bucket_3way_derived"]["per_bucket"]
    for b in LABELS:
        r = rep[b]
        print(f"  {b:8s} precision={r['precision']:.3f}  recall={r['recall']:.3f}  f1={r['f1-score']:.3f}")
    print(f"  macro F1 = {rep['macro avg']['f1-score']:.4f}")
    print("Confusion matrix (rows=true, cols=pred):", results["bucket_3way_derived"]["confusion_matrix"])
    t = results["triage"]
    print(f"Triage: cutoff={t['cutoff_chosen_on_validation']:.4f}  test recall={t['test_recall_at_cutoff']:.4f}  "
          f"precision={t['test_precision_at_cutoff']:.4f}  flag rate={t['test_flag_rate']:.1%}  "
          f"serious missed={t['serious_missed_at_cutoff']}")
    print(f"        AUPRC={t['auprc']:.4f}  AUROC={t['auroc']:.4f}  "
          f"Brier raw={t['brier_score_raw']:.4f} calibrated={t['brier_score_calibrated']:.4f}")
    print(f"        sensitivity at {SPONSOR_SPECIFICITY:.0%} specificity: "
          f"{t[f'sensitivity_at_{SPONSOR_SPECIFICITY}_specificity']:.4f}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--target", choices=sorted(TARGETS), default="bucket")
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    parser.add_argument("--epochs", type=float, default=3)
    args = parser.parse_args(argv)

    run_name = f"{args.target}_len{args.max_length}"
    out_dir = os.path.join(OUT_DIR, run_name)
    label_col, class_names = TARGETS[args.target]
    n_classes = len(class_names)

    train_df, val_df, test_df = train_val_test()
    print(f"Run {run_name}. Train:", train_df.shape, "Val:", val_df.shape, "Test:", test_df.shape)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("Using device:", device)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length, padding=False)

    def to_dataset(df):
        ds = Dataset.from_pandas(df[["text", label_col]].rename(columns={label_col: "labels"}).reset_index(drop=True))
        ds = ds.map(tokenize_fn, batched=True)
        ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        return ds

    train_ds, val_ds, test_ds = to_dataset(train_df), to_dataset(val_df), to_dataset(test_df)
    truncated = np.mean([len(x) >= args.max_length for x in train_ds["input_ids"]])
    print(f"Train reports truncated at {args.max_length} tokens: {truncated:.1%}")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=n_classes,
        id2label=dict(enumerate(class_names)), label2id={l: i for i, l in enumerate(class_names)},
    )

    # class weights, same idea as the baseline's class_weight="balanced"
    counts = train_df[label_col].value_counts().reindex(range(n_classes), fill_value=0).to_numpy(dtype=np.float32)
    inv_freq = 1.0 / np.where(counts == 0, 1e-6, counts)
    class_weights = inv_freq / inv_freq.sum() * n_classes
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
    print("Class weights:", dict(zip(class_names, np.round(class_weights, 4).tolist())))

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
        # Checkpoint selection always scores the 3-bucket view (derived, for
        # --target grade), so both targets pick checkpoints the same way.
        logits, labels = eval_pred
        proba = softmax(logits, axis=1)
        preds = to_bucket_probs(proba, args.target).argmax(axis=1)
        bucket_labels = labels if args.target == "bucket" else GRADE_TO_BUCKET[labels]
        report = classification_report(bucket_labels, preds, labels=list(range(len(LABELS))),
                                       target_names=LABELS, output_dict=True, zero_division=0)
        return {
            "accuracy": report["accuracy"],
            "serious_recall": report["serious"]["recall"],
            "serious_precision": report["serious"]["precision"],
            "macro_f1": report["macro avg"]["f1-score"],
            "triage_auprc": average_precision_score(bucket_labels > 0, to_triage(proba, args.target)),
        }

    training_args = TrainingArguments(
        output_dir=out_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        num_train_epochs=args.epochs,
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

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        class_weights=class_weights_tensor,
    )

    trainer.train()

    proba_val = softmax(trainer.predict(val_ds).predictions, axis=1)
    proba_test = softmax(trainer.predict(test_ds).predictions, axis=1)
    np.save(os.path.join(out_dir, "proba_val.npy"), proba_val)
    np.save(os.path.join(out_dir, "proba_test.npy"), proba_test)

    results = evaluate(args.target, proba_val, proba_test, val_df, test_df)
    results["config"] = {
        "model": MODEL_NAME, "target": args.target, "max_length": args.max_length,
        "epochs": args.epochs, "train_truncated_share": float(truncated),
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "epoch_log": [h for h in trainer.state.log_history if "eval_serious_recall" in h],
    }

    print(f"\n=== DistilBERT ({run_name}) on held-out test set ===")
    print_summary(results)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"{run_name}.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print("\nSaved results to", path)


if __name__ == "__main__":
    main()
