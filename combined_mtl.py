"""
Combines this project's 3-bucket severity task with the teammate's 11-buckets
event-type task in one shared model, to test a specific hypothesis: does
knowing the predicted event type (fall, medication error, etc.) help predict
harm severity, especially for the rare `serious` class (359 train / 20 test
rows)?

Architecture mirrors the teammate's `11-buckets/src/mtl.py`:
  - Frozen sentence-transformers/all-MiniLM-L6-v2 encoder (same model, same
    reason: fast enough to iterate on a laptop, no GPU needed).
  - Head A: 11-way event type, trained on the embedding.
  - Head C (this project's contribution): 3-way severity bucket, trained on
    the embedding PLUS Head A's 11 softmax probabilities -- same "clue"
    pattern as their Head B (hurt/not-hurt), just predicting severity instead
    of/alongside binary hurt.
  - An ablation Head C' that gets the embedding only, no event-type clue, so
    the effect of the clue can be measured directly instead of assumed.
  - Both heads trained together with one combined loss (event-type loss +
    severity loss, equal weights), across multiple random seeds -- same
    reasoning as their 5-seed baseline: on a class this rare, a single run's
    number is not distinguishable from seed noise.

Data: data_pipeline.py (leakage-safe fields, real Jan-Aug/Sep/Oct split, now
extended with the 11-bucket event_type_label alongside the 3-bucket label).

Run: python3 combined_mtl.py
"""

import json
import time

import numpy as np
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from sklearn.metrics import classification_report, accuracy_score

from data_pipeline import train_val_test, LABELS, N_EVENT_CLASSES, UNLABELED

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384
HIDDEN = 256
DROPOUT = 0.2
SEEDS = [0, 1, 2]
EPOCHS = 30
BATCH_SIZE = 128
LR = 1e-3
CACHE = {"train": "./_mtl_embed_train.npy", "val": "./_mtl_embed_val.npy", "test": "./_mtl_embed_test.npy"}


def embed_texts(model, texts, device):
    # normalize_embeddings=True: unit-norm vectors train much more stably with
    # a small MLP head than raw MiniLM output, which has uneven magnitude.
    return model.encode(
        list(texts), batch_size=128, show_progress_bar=True, convert_to_numpy=True,
        device=device, normalize_embeddings=True,
    ).astype(np.float32)


def get_embeddings(train_df, val_df, test_df, device):
    import os

    if all(os.path.exists(p) for p in CACHE.values()):
        print("Loading cached MiniLM embeddings from disk...")
        return (np.load(CACHE["train"]), np.load(CACHE["val"]), np.load(CACHE["test"]))

    print(f"Embedding with {EMBED_MODEL} on {device} (one-time, cached to disk after)...")
    model = SentenceTransformer(EMBED_MODEL, device=device)
    t0 = time.time()
    emb_train = embed_texts(model, train_df["text"], device)
    emb_val = embed_texts(model, val_df["text"], device)
    emb_test = embed_texts(model, test_df["text"], device)
    print(f"Embedding took {time.time() - t0:.1f}s")
    np.save(CACHE["train"], emb_train)
    np.save(CACHE["val"], emb_val)
    np.save(CACHE["test"], emb_test)
    return emb_train, emb_val, emb_test


class MTLHeads(nn.Module):
    """Head A (event type) -> Head C (severity), with an ablation switch."""

    def __init__(self, embed_dim, n_event_classes, n_severity_classes, hidden, dropout, use_clue):
        super().__init__()
        self.use_clue = use_clue
        self.head_a = nn.Sequential(
            nn.Linear(embed_dim, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, n_event_classes)
        )
        head_c_in = embed_dim + n_event_classes if use_clue else embed_dim
        self.head_c = nn.Sequential(
            nn.Linear(head_c_in, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, n_severity_classes)
        )

    def forward(self, x):
        logits_a = self.head_a(x)
        if self.use_clue:
            clue = torch.softmax(logits_a.detach(), dim=-1)  # detached: clue informs, doesn't backprop through A
            logits_c = self.head_c(torch.cat([x, clue], dim=-1))
        else:
            logits_c = self.head_c(x)
        return logits_a, logits_c


def class_weights(labels, n_classes, unlabeled=None):
    labels = labels[labels != unlabeled] if unlabeled is not None else labels
    counts = np.bincount(labels, minlength=n_classes).astype(np.float32)
    inv = 1.0 / np.where(counts == 0, 1e-6, counts)
    return torch.tensor(inv / inv.sum() * n_classes, dtype=torch.float32)


def train_one_run(emb_train, emb_val, y_event_train, y_sev_train, use_clue, seed, device):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = MTLHeads(EMBED_DIM, N_EVENT_CLASSES, len(LABELS), HIDDEN, DROPOUT, use_clue).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    x_all = torch.tensor(emb_train, dtype=torch.float32, device=device)
    y_event_all = torch.tensor(y_event_train, dtype=torch.long, device=device)
    y_sev_all = torch.tensor(y_sev_train, dtype=torch.long, device=device)
    event_mask_all = y_event_all != UNLABELED

    w_event = class_weights(y_event_train, N_EVENT_CLASSES, unlabeled=UNLABELED).to(device)
    w_sev = class_weights(y_sev_train, len(LABELS)).to(device)

    n = x_all.shape[0]
    rng = np.random.RandomState(seed)

    for epoch in range(EPOCHS):
        model.train()
        perm = rng.permutation(n)
        for start in range(0, n, BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            x = x_all[idx]
            y_event = y_event_all[idx]
            y_sev = y_sev_all[idx]
            mask = event_mask_all[idx]

            opt.zero_grad()
            logits_a, logits_c = model(x)
            loss_c = nn.functional.cross_entropy(logits_c, y_sev, weight=w_sev)
            if mask.any():
                loss_a = nn.functional.cross_entropy(logits_a[mask], y_event[mask], weight=w_event)
                loss = loss_a + loss_c
            else:
                loss = loss_c
            loss.backward()
            opt.step()

    return model


def evaluate(model, emb, y_event, y_sev, device):
    model.eval()
    x = torch.tensor(emb, dtype=torch.float32, device=device)
    with torch.no_grad():
        logits_a, logits_c = model(x)
        pred_a = logits_a.argmax(dim=-1).cpu().numpy()
        pred_c = logits_c.argmax(dim=-1).cpu().numpy()

    event_mask = y_event != UNLABELED
    event_acc = accuracy_score(y_event[event_mask], pred_a[event_mask])
    sev_report = classification_report(
        y_sev, pred_c, target_names=LABELS, output_dict=True, zero_division=0
    )
    return event_acc, sev_report


def summarize(reports, key_metric_path):
    vals = []
    for r in reports:
        v = r
        for k in key_metric_path:
            v = v[k]
        vals.append(v)
    return float(np.mean(vals)), float(np.std(vals))


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("Device for embedding/training:", device)

    train_df, val_df, test_df = train_val_test()
    print("Train:", train_df.shape, "Val:", val_df.shape, "Test:", test_df.shape)

    emb_train, emb_val, emb_test = get_embeddings(train_df, val_df, test_df, device)

    y_event_train = train_df["event_type_label"].to_numpy()
    y_sev_train = train_df["label"].to_numpy()
    y_event_test = test_df["event_type_label"].to_numpy()
    y_sev_test = test_df["label"].to_numpy()

    results = {"with_clue": [], "no_clue": []}
    for use_clue, key in [(True, "with_clue"), (False, "no_clue")]:
        print(f"\n=== {'Severity head WITH event-type clue' if use_clue else 'Severity head WITHOUT clue (ablation)'} ===")
        for seed in SEEDS:
            model = train_one_run(emb_train, emb_val, y_event_train, y_sev_train, use_clue, seed, device)
            event_acc, sev_report = evaluate(model, emb_test, y_event_test, y_sev_test, device)
            print(
                f"seed={seed}  event_acc={event_acc:.3f}  "
                f"serious: P={sev_report['serious']['precision']:.3f} R={sev_report['serious']['recall']:.3f}  "
                f"some: P={sev_report['some']['precision']:.3f} R={sev_report['some']['recall']:.3f}  "
                f"macro_f1={sev_report['macro avg']['f1-score']:.3f}"
            )
            results[key].append({"seed": seed, "event_acc": event_acc, "sev_report": sev_report})

    print("\n" + "=" * 70)
    print("SUMMARY (mean +/- std over seeds, test set)")
    print("=" * 70)
    for key, label in [("with_clue", "WITH event-type clue"), ("no_clue", "WITHOUT clue (ablation)")]:
        reports = [r["sev_report"] for r in results[key]]
        event_accs = [r["event_acc"] for r in results[key]]
        serious_p = summarize(reports, ["serious", "precision"])
        serious_r = summarize(reports, ["serious", "recall"])
        some_p = summarize(reports, ["some", "precision"])
        some_r = summarize(reports, ["some", "recall"])
        macro_f1 = summarize(reports, ["macro avg", "f1-score"])
        print(f"\n{label}:")
        print(f"  event-type accuracy:   {np.mean(event_accs):.3f} +/- {np.std(event_accs):.3f}")
        print(f"  serious precision:     {serious_p[0]:.3f} +/- {serious_p[1]:.3f}")
        print(f"  serious recall:        {serious_r[0]:.3f} +/- {serious_r[1]:.3f}")
        print(f"  some precision:        {some_p[0]:.3f} +/- {some_p[1]:.3f}")
        print(f"  some recall:           {some_r[0]:.3f} +/- {some_r[1]:.3f}")
        print(f"  macro F1:              {macro_f1[0]:.3f} +/- {macro_f1[1]:.3f}")

    with open("./combined_mtl_results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    print("\nSaved full results to ./combined_mtl_results.json")


if __name__ == "__main__":
    main()
