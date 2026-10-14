"""
The combined event-type + severity model again, this time on a fine-tuned
encoder instead of frozen MiniLM embeddings (experiments/combined_mtl.py).
The frozen version was too weak to say whether knowing the event type helps
predict harm; this asks the same question of a model strong enough to answer.

Architecture: DistilBERT, fine-tuned end to end, two heads on the first-token
hidden state:
  - event head: 11-way event type (the teammate's 11-buckets labels; the ~64
    reports with no event code are skipped by this head's loss only)
  - severity head: all 10 PSRS grades, fed the hidden state PLUS the event
    head's softmax probabilities (detached, as in combined_mtl.py: the clue
    informs severity but severity loss doesn't train the event head)
  - --no-clue: same joint training, severity head sees the hidden state only.
    The clue's effect is the difference between the two runs.
Loss = event loss + severity loss, equal weights, both class-weighted.

The single-task baseline is experiments/finetune_distilbert.py --target grade at the same
--max-length. Severity outputs are scored by finetune_distilbert.evaluate()
(bucket and triage derived the production way, cutoff and calibration fit on
September, everything reported on October), so the three runs compare
directly. The best epoch is chosen the same way as there: derived serious
recall on September.

Run from the repo root:
    python3 -m experiments.combined_finetune               # with the clue
    python3 -m experiments.combined_finetune --no-clue     # ablation
    python3 -m experiments.combined_finetune --limit 500 --epochs 1   # smoke test

Summary JSON goes to distilbert_results/combined_<clue|noclue>_len<N>.json;
probabilities and weights go to distilbert_output/ (gitignored).
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from scipy.special import softmax
from sklearn.metrics import accuracy_score, classification_report
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from data_pipeline import EVENT_BUCKETS, FULL_LABELS, LABELS, N_EVENT_CLASSES, UNLABELED, train_val_test
from experiments.finetune_distilbert import GRADE_TO_BUCKET, MODEL_NAME, OUT_DIR, RESULTS_DIR, evaluate, print_summary

N_GRADES = len(FULL_LABELS)


class JointModel(nn.Module):
    def __init__(self, use_clue):
        super().__init__()
        self.use_clue = use_clue
        self.encoder = AutoModel.from_pretrained(MODEL_NAME)
        hidden = self.encoder.config.dim
        self.dropout = nn.Dropout(0.1)
        self.event_head = nn.Linear(hidden, N_EVENT_CLASSES)
        self.severity_head = nn.Linear(hidden + (N_EVENT_CLASSES if use_clue else 0), N_GRADES)

    def forward(self, input_ids, attention_mask):
        h = self.dropout(self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0])
        event_logits = self.event_head(h)
        if self.use_clue:
            h = torch.cat([h, torch.softmax(event_logits.detach(), dim=-1)], dim=-1)
        return event_logits, self.severity_head(h)


def class_weights(labels, n_classes):
    """Same inverse-frequency weights as finetune_distilbert.py; unlabeled rows skipped."""
    counts = np.bincount(labels[labels != UNLABELED], minlength=n_classes).astype(np.float32)
    inv = 1.0 / np.where(counts == 0, 1e-6, counts)
    return torch.tensor(inv / inv.sum() * n_classes, dtype=torch.float32)


def batches(df, tokenizer, max_length, batch_size, shuffle, rng=None):
    order = rng.permutation(len(df)) if shuffle else np.arange(len(df))
    for i in range(0, len(df), batch_size):
        rows = df.iloc[order[i : i + batch_size]]
        enc = tokenizer(list(rows["text"]), truncation=True, max_length=max_length,
                        padding=True, return_tensors="pt")
        yield (enc["input_ids"], enc["attention_mask"],
               torch.tensor(rows["event_type_label"].to_numpy()), torch.tensor(rows["full_label"].to_numpy()))


@torch.no_grad()
def predict(model, df, tokenizer, max_length, device):
    model.eval()
    ev, sev = [], []
    for ids, mask, _, _ in batches(df, tokenizer, max_length, 64, shuffle=False):
        e, s = model(ids.to(device), mask.to(device))
        ev.append(e.float().cpu().numpy())
        sev.append(s.float().cpu().numpy())
    return softmax(np.concatenate(ev), axis=1), softmax(np.concatenate(sev), axis=1)


def serious_recall(proba_grade, df):
    """Checkpoint-selection metric, matching finetune_distilbert.compute_metrics."""
    y_bucket = GRADE_TO_BUCKET[df["full_label"].to_numpy()]
    pred_bucket = GRADE_TO_BUCKET[proba_grade.argmax(axis=1)]
    serious = LABELS.index("serious")
    hits = (y_bucket == serious) & (pred_bucket == serious)
    return hits.sum() / max((y_bucket == serious).sum(), 1)


def event_metrics(proba_event, df):
    y = df["event_type_label"].to_numpy()
    keep = y != UNLABELED
    pred = proba_event.argmax(axis=1)
    rep = classification_report(y[keep], pred[keep], labels=list(range(N_EVENT_CLASSES)),
                                target_names=list(EVENT_BUCKETS), output_dict=True, zero_division=0)
    return {"accuracy": accuracy_score(y[keep], pred[keep]), "macro_f1": rep["macro avg"]["f1-score"],
            "n_labeled": int(keep.sum())}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--no-clue", action="store_true", help="severity head does not see event-type probabilities")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None, help="use only the first N rows of each split (smoke test)")
    args = parser.parse_args(argv)

    use_clue = not args.no_clue
    run_name = f"combined_{'clue' if use_clue else 'noclue'}_len{args.max_length}"
    if args.limit:
        run_name += f"_limit{args.limit}"
    out_dir = os.path.join(OUT_DIR, run_name)
    os.makedirs(out_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    train_df, val_df, test_df = train_val_test()
    if args.limit:
        train_df, val_df, test_df = (d.head(args.limit) for d in (train_df, val_df, test_df))
    print(f"Run {run_name}. Train:", train_df.shape, "Val:", val_df.shape, "Test:", test_df.shape)

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = JointModel(use_clue).to(device)
    event_loss = nn.CrossEntropyLoss(
        weight=class_weights(train_df["event_type_label"].to_numpy(), N_EVENT_CLASSES).to(device),
        ignore_index=UNLABELED)
    severity_loss = nn.CrossEntropyLoss(
        weight=class_weights(train_df["full_label"].to_numpy(), N_GRADES).to(device))

    steps_per_epoch = -(-len(train_df) // args.batch_size)
    no_decay = ("bias", "LayerNorm.weight")
    params = [
        {"params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)], "weight_decay": 0.01},
        {"params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)], "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(params, lr=args.lr)
    sched = get_linear_schedule_with_warmup(opt, 0, steps_per_epoch * args.epochs)

    best = {"serious_recall": -1.0}
    epoch_log = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, running = time.time(), 0.0
        for step, (ids, mask, y_event, y_grade) in enumerate(
                batches(train_df, tokenizer, args.max_length, args.batch_size, shuffle=True, rng=rng), 1):
            e, s = model(ids.to(device), mask.to(device))
            y_event = y_event.to(device)
            loss = severity_loss(s, y_grade.to(device))
            if (y_event != UNLABELED).any():
                loss = loss + event_loss(e, y_event)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            running += loss.item()
            if step % 100 == 0:
                print(f"  epoch {epoch} step {step}/{steps_per_epoch} loss {running / 100:.4f} "
                      f"({time.time() - t0:.0f}s)", flush=True)
                running = 0.0

        _, proba_val_grade = predict(model, val_df, tokenizer, args.max_length, device)
        rec = serious_recall(proba_val_grade, val_df)
        epoch_log.append({"epoch": epoch, "val_serious_recall": rec, "seconds": time.time() - t0})
        print(f"Epoch {epoch}: val derived serious recall {rec:.3f} ({time.time() - t0:.0f}s)", flush=True)
        if rec > best["serious_recall"]:
            best = {"serious_recall": rec, "epoch": epoch}
            torch.save(model.state_dict(), os.path.join(out_dir, "best.pt"))

    model.load_state_dict(torch.load(os.path.join(out_dir, "best.pt"), map_location=device))
    proba_val_event, proba_val = predict(model, val_df, tokenizer, args.max_length, device)
    proba_test_event, proba_test = predict(model, test_df, tokenizer, args.max_length, device)
    for name, arr in [("proba_val", proba_val), ("proba_test", proba_test),
                      ("proba_val_event", proba_val_event), ("proba_test_event", proba_test_event)]:
        np.save(os.path.join(out_dir, f"{name}.npy"), arr)

    results = evaluate("grade", proba_val, proba_test, val_df, test_df)
    results["event_type"] = {"validation": event_metrics(proba_val_event, val_df),
                             "test": event_metrics(proba_test_event, test_df)}
    results["config"] = {
        "model": MODEL_NAME, "use_event_clue": use_clue, "max_length": args.max_length,
        "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr, "seed": args.seed,
        "limit": args.limit, "best_epoch": best["epoch"], "epoch_log": epoch_log,
    }

    print(f"\n=== Combined DistilBERT ({run_name}) on held-out test set ===")
    print_summary(results)
    et = results["event_type"]["test"]
    print(f"Event type: accuracy={et['accuracy']:.4f}  macro F1={et['macro_f1']:.4f}  (n={et['n_labeled']})")

    results_dir = out_dir if args.limit else RESULTS_DIR  # smoke tests stay out of the committed results
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, f"{run_name}.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print("\nSaved results to", path)


if __name__ == "__main__":
    main()
