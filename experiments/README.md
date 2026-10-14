# Experiments

Retired and one-off scripts, kept for history and for comparison runs. None
of them are part of the pipeline (`data_pipeline.py`, `severity_model.py`,
`severity_model_extended.py`, `rolling_evaluation.py`, `predict.py` in the
repo root). Results and conclusions are written up in
[PIPELINE.md](../PIPELINE.md).

| Script | What it tests | Status |
|---|---|---|
| `baseline_grader.py` | TF-IDF + LogReg trained directly on the 3 buckets | Superseded by the 10-grade model |
| `ordinal_vs_coarse.py` | 10-grade model vs. a model trained only on hurt / not hurt | 10-grade model kept; became `severity_model.py` |
| `augment_rare_grades.py` | Hand-written extra training reports for grades G/H/I | Rejected, no measurable effect |
| `combined_mtl.py` | Feeding a predicted event type into severity (frozen MiniLM) | Rejected: encoder too weak to answer the question |
| `combined_finetune.py` | The same idea on fine-tuned DistilBERT (`--no-clue` = ablation) | Written and smoke-tested; full runs not done yet |
| `finetune_distilbert.py` | DistilBERT, `--target bucket` or `--target grade`, `--max-length N` | 3-bucket at 128 tokens: macro F1 0.79 vs. production's 0.82. 10-grade at 256 not run yet |
| `error_analysis.py` | Lists the harmed October reports the production model doesn't flag | Current (uses the saved production model) |
| `leakage_inflation_check.py` | Effect of adding the post-review fields back into the input | Little effect on this synthetic data |
| `longformer_colab.ipynb` | Clinical-Longformer on the 3 buckets, for a Colab GPU | Uses the leakage-safe date split; not run yet |

Run them as modules from the repo root, so they can import `data_pipeline`
and their relative output paths resolve there:

```bash
python3 -m experiments.error_analysis
python3 -m experiments.finetune_distilbert --target grade --max-length 256
```

The DistilBERT scripts take hours on a laptop GPU (about 3 s per training
step at 256 tokens on an Apple M-series). `leakage_inflation_check.py` reads
the excluded columns on purpose, to measure their effect.
