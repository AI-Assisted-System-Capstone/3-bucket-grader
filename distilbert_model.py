"""
Save, load and score a fine-tuned DistilBERT severity model.

experiments/finetune_distilbert.py (--target grade) writes a folder with
save_distilbert(); predict.py --model distilbert reads it back with
load_distilbert(). The folder holds everything scoring needs:

    model/            the fine-tuned weights (Hugging Face format)
    tokenizer/        the tokenizer the weights were trained with
    calibrator.joblib Platt scaling fit on September
    config.json       triage cutoff frozen on September, max_length, target

Only a 10-grade model can be saved here: predict.py reports the letter grade
and derives the buckets and triage score from grade probabilities, exactly as
the word-counting model does. torch and transformers are imported inside the
functions so predict.py still runs without them for the default model.
"""

import json
import os
from dataclasses import dataclass

import joblib
import numpy as np

from data_pipeline import FULL_LABELS

BATCH_SIZE = 64


@dataclass
class DistilBertModel:
    tokenizer: object
    network: object
    calibrator: object
    cutoff: float
    max_length: int
    classes_: np.ndarray

    def predict_proba(self, texts):
        """Grade probabilities, shape (n, 10), columns in `classes_` order."""
        import torch

        texts = list(texts)
        device = next(self.network.parameters()).device
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), BATCH_SIZE):
                enc = self.tokenizer(
                    texts[i:i + BATCH_SIZE], truncation=True, max_length=self.max_length,
                    padding=True, return_tensors="pt",
                ).to(device)
                logits = self.network(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits
                out.append(torch.softmax(logits, dim=-1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, len(self.classes_)))


def save_distilbert(out_dir, trainer, tokenizer, calibrator, cutoff, max_length, target_recall):
    """Write a folder load_distilbert() can read. `trainer` must hold the best checkpoint."""
    os.makedirs(out_dir, exist_ok=True)
    trainer.save_model(os.path.join(out_dir, "model"))
    tokenizer.save_pretrained(os.path.join(out_dir, "tokenizer"))
    joblib.dump(calibrator, os.path.join(out_dir, "calibrator.joblib"))
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump({
            "target": "grade",
            "cutoff_raw_triage": float(cutoff),
            "target_recall": target_recall,
            "max_length": int(max_length),
        }, f, indent=2)


def load_distilbert(model_dir):
    needed = [os.path.join(model_dir, p) for p in ("model", "tokenizer", "calibrator.joblib", "config.json")]
    missing = [p for p in needed if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            f"DistilBERT artifacts missing: {missing}. distilbert_output/ is gitignored -- run "
            f"`python3 -m experiments.finetune_distilbert --target grade --max-length 256` first."
        )
    with open(os.path.join(model_dir, "config.json")) as f:
        cfg = json.load(f)
    if cfg.get("target") != "grade":
        raise ValueError(f"{model_dir} holds a {cfg.get('target')!r}-target model; predict.py needs --target grade")

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    network = AutoModelForSequenceClassification.from_pretrained(os.path.join(model_dir, "model")).eval()
    if network.config.num_labels != len(FULL_LABELS):
        raise ValueError(f"{model_dir} has {network.config.num_labels} classes; expected {len(FULL_LABELS)} grades")
    return DistilBertModel(
        tokenizer=AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer")),
        network=network,
        calibrator=joblib.load(os.path.join(model_dir, "calibrator.joblib")),
        cutoff=float(cfg["cutoff_raw_triage"]),
        max_length=int(cfg["max_length"]),
        classes_=np.arange(len(FULL_LABELS)),
    )
