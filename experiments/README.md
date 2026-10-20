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
| `combined_finetune.py` | The same idea on fine-tuned DistilBERT (`--no-clue` = ablation) | Done: 93.8% caught with or without the event-type hint, so the hint didn't help |
| `finetune_distilbert.py` | DistilBERT, `--target bucket` or `--target grade`, `--max-length N`, `--seed N` (default 42); `--target grade` also saves a model `predict.py --model distilbert` loads | 10 grades at 256 tokens: 94.3% caught at 96% specificity vs 90.3% for production. Candidate to replace it |
| `error_analysis.py` | Lists the harmed October reports the production model doesn't flag | Current (uses the saved production model) |
| `leakage_inflation_check.py` | Effect of adding the post-review fields back into the input | Little effect on this synthetic data |
| `longformer_colab.ipynb` | Clinical-Longformer on the 3 buckets, for a Colab GPU | Superseded by the head-to-head run of the sponsor's own script (results in `sponsor_comparison/`) |
| `improve_sensitivity.py` | 11 settings of the word-counting model, chosen on June–September | Best (letter chunks, C=4): 91.6% on October vs 90.3% |
| `sponsor_comparison/` | Results of the sponsor's Clinical-Longformer on our data and split | 91.6% on the report as submitted, 93.8% with manager comments and unit actions |

Run them as modules from the repo root, so they can import `data_pipeline`
and their relative output paths resolve there:

```bash
python3 -m experiments.error_analysis
python3 -m experiments.finetune_distilbert --target grade --max-length 256
```

The DistilBERT scripts take hours on a laptop GPU (about 3 s per training
step at 256 tokens on an Apple M-series). `leakage_inflation_check.py` reads
the excluded columns on purpose, to measure their effect.
