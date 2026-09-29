# Harm Severity Grader

Grades free-text patient safety incident reports on the 10-grade PSRS harm
scale (A through I) so reviewers can triage the highest-risk reports first
instead of reading every one. One model produces three outputs:

- **Letter grade** (A, B1, B2, C, D, E, F, G, H, I)
- **3-bucket label**: `none` / `some` / `serious`
- **Triage score**: a calibrated probability that the patient was harmed,
  plus a flag for priority review

This repo covers the severity/triage requirement. The teammate's `11-buckets`
repo covers event-type and recurring-theme classification. The two answer
different sponsor requirements and stay separate.

See [PIPELINE.md](PIPELINE.md) for diagrams and full per-experiment detail.

## Data

Synthetic MIDAS-style dataset: 80,000 reports, 23 columns. 67,003 are usable
after dropping rows with no harm score or no narrative text.

| Bucket | Grades | Meaning |
|---|---|---|
| `none` | A, B1, B2, C, D | Unsafe condition, near miss, or no harm |
| `some` | E, F | Temporary harm, treated (incl. added hospitalization) |
| `serious` | G, H, I | Permanent harm, near death, or death |

**Split by date**, decoded from the `Event No.` prefix, so the model is always
tested on reports from a later month than it trained on:

| Bucket | Train (Jan–Aug) | Val (Sep) | Test (Oct) |
|---|---|---|---|
| `none` | 42,523 (84.9%) | 8,245 (95.9%) | 7,909 (95.2%) |
| `some` | 7,209 (14.4%) | 339 (3.9%) | 383 (4.6%) |
| `serious` | 359 (0.7%) | 16 (0.2%) | 20 (0.2%) |

Training over-samples harm cases on purpose; validation and test reflect the
natural rate.

## Model inputs (leakage-safe)

The model only reads what's available when a report is submitted:

- `event_comments`, the reporter's narrative
- A short labeled prefix of intake fields: unit (`Location Name`), service
  (`Encounter Service`), age band, and prescribed / administered / suspect
  medication and dose

These fields are **never** used, because they're filled in after a safety
officer reviews the report and would leak the answer:
`manager_comments`, `unit_actions_taken`, `shareable_lessons`, and
`Analyst-Report Type`. `data_pipeline.py` asserts on every load that the
first three never reach the model.

## Workflow

```
data_pipeline.py            load → drop leaked fields → build input text → date split → derive labels
        │
severity_model.py           train TF-IDF + LogReg on 10 grades → derive bucket + triage score
        │                   → pick cutoff on Sep → calibrate on Sep → evaluate on Oct → save
        │
severity_model_extended.py  sponsor comparison, operating-point menu, per-day recall@k,
        │                   subgroup breakdown with confidence intervals, explanations
        │
prefill_extra_fields.py     HPI Designation classifier (pre-fill helper)
```

### How the model works

1. **Text to features:** TF-IDF on unigrams and bigrams, top 20,000 features,
   English stop words removed. It uses a custom token pattern so hyphenated
   terms like "X-ray" stay intact.
2. **Classifier:** multinomial logistic regression on all 10 grades, with
   `class_weight="balanced"` to handle the heavy imbalance.
3. **Derived outputs:** the 3-bucket label sums each group's grade
   probabilities. The triage score is the summed probability of grades E–I.
4. **Cutoff:** picked on September (validation) as the highest cutoff that
   still reaches ≥95% recall on harm cases, then frozen and applied to
   October unchanged. The 95% target is this project's own default, not a
   documented sponsor requirement.
5. **Calibration:** Platt scaling fit on validation, so the triage score reads
   as a real probability despite training's inflated harm rate.

### Running it

Scripts need Python 3.11 with pandas, scikit-learn and joblib (transformer
experiments also need torch, transformers, datasets and accelerate). The
dataset isn't in the repo; point `DATA_PATH` in `data_pipeline.py` at your
copy of the parquet file.

```bash
python3 severity_model.py            # trains, evaluates, saves to severity_model/
python3 severity_model_extended.py   # extended evaluation
python3 prefill_extra_fields.py      # HPI Designation classifier
```

Outputs in `severity_model/`: `evaluation.json`, `extended_evaluation.json`
and `config.json` are committed; the `.joblib` model files are gitignored.

## Results

All numbers are on the October test set (8,312 reports), from
`severity_model/evaluation.json` and `extended_evaluation.json`.

### Triage (harmed vs. not harmed)

At the production cutoff (0.236):

| Metric | Value |
|---|---|
| Recall | **96.0%** |
| Precision | 28.9% |

### Letter grade (10-way)

Exact accuracy 76.6%, mean absolute error 0.39 grades, quadratic-weighted
kappa 0.72.

### 3-bucket (derived)

| Bucket | Precision | Recall | Test cases |
|---|---|---|---|
| `none` | 0.99 | 0.99 | 7,909 |
| `some` | 0.76 | 0.83 | 383 |
| `serious` | 0.62 | 0.75 | 20 |

Macro F1: 0.82.

### Compared with the sponsor's model

The sponsor reported sensitivity (the same thing as recall) at a fixed
specificity: the share of unharmed reports correctly *not* flagged. Setting
this model to the same 96.0% specificity:

| | Sensitivity (recall) | Specificity |
|---|---|---|
| Sponsor's Clinical-Longformer | 95.2% | 96.0% |
| This model | 90.3% | 96.0% |

This is not a like-for-like comparison. The sponsor's script feeds
`MANAGER COMMENTS` and `UNIT_ACTIONS_TAKEN` into the model, uses a random
rather than date-based split, and was measured on real data, not this
synthetic set.

## Known issues and limitations

- **Some serious cases are missed.** 2–3 of the 20 G/H/I test cases fall below
  the production cutoff. Their narratives use hedged, near-miss language
  ("could have received an incorrect dose") rather than outcome words.
  A cutoff of about 0.152 catches all of them, at roughly 25 more false
  alarms per day. That cutoff was fit on test, so it's optimistic.
- **Validation can't tune for those misses.** Every September G/H/I case
  scores ≥0.350, while October's misses score 0.152–0.310.
- **The serious tier is tiny.** 20 test cases (6 G, 14 H, 0 I) and 16 validation
  cases, so `serious` metrics have wide confidence intervals and grade I isn't
  evaluated at all.
- **Rare grades are weak on their own.** B1 F1 is 0.21 and G F1 is 0.55. The
  letter-grade macro F1 is 0.59, versus 0.82 once grades are grouped into
  buckets.
- **Precision is low at the default cutoff.** About 7 in 10 flagged reports are
  false alarms. That's the cost of the 95% recall target.
- **Narrative/label mismatch.** Many false negatives say "no harm noted" while
  carrying a harm grade. This could be real, or noise from the synthetic data.
- **Synthetic data only.** No real-data validation. Public alternatives were
  checked: NRLS/LFPSE and SAFRON are access-gated, FAERS has no narrative text,
  and IFMIR has no severity labels.
- **No template-overlap check.** There's no archetype ID column, so near-duplicate
  templates across train and test can't be ruled out.
- **Subgroup results are inconclusive.** Recall intervals overlap across all
  services, and subgroup membership is generator-assigned, not real.
- **Some fields aren't modeled.** Level of Investigation isn't modeled because 98%
  of rows have one value, which leaves too few RCA cases.

## Open tasks

### Code improvements

- [ ] **Add a prediction entry point.** The only way to score new reports is
      `demo_predict()` in `severity_model.py`, which runs one hardcoded
      example. Add a `predict.py` that loads `severity_model/` and scores a
      CSV/parquet file, writing out grade, bucket, calibrated triage
      probability and the flag.
- [ ] **Add tests for the safety-critical pieces.** Small `pytest` checks for
      the excluded-column assertion, `Event No.` split decoding, the grade →
      bucket mapping, `bucket_probs` summing to 1, and a round-trip load of
      the saved model.
- [ ] **Separate production code from experiments.** Move `baseline_grader.py`,
      `ordinal_vs_coarse.py`, `combined_mtl.py`, `augment_rare_grades.py` and
      `error_analysis.py` into `experiments/`, so it's obvious that
      `data_pipeline.py`, `severity_model.py` and `severity_model_extended.py`
      are the pipeline.
- [ ] **Point `error_analysis.py` at the production model.** It was written
      for the 3-bucket DistilBERT output. Running it on the 10-grade model
      would make the false-negative review repeatable on what's shipped.

### Experiments

- [ ] **Tune DistilBERT's `MAX_LENGTH`.** 128 tokens was a guess, never
      checked against the dataset's token-length distribution. The 0.79
      result likely undersells it.
- [ ] **Train DistilBERT on all 10 grades.** It's only been run on 3 buckets.
      Deriving bucket and triage outputs the same way as production would give a
      like-for-like comparison. Its better `serious` precision makes this worth
      doing.
- [ ] **Run the sponsor's Clinical-Longformer script head-to-head.** Needs one
      GPU/Colab session. Run it twice on the date split: once as written, and
      once with `MANAGER COMMENTS` / `UNIT_ACTIONS_TAKEN` removed. It will
      need updates for current `transformers`:
      - `compute_loss` needs `num_items_in_batch=None`
      - `evaluation_strategy` → `eval_strategy`
      - `tokenizer=` → `processing_class=`

      This measures the leakage effect on synthetic data only.
- [ ] **Retry the combined model on a fine-tuned encoder.** Use DistilBERT or
      Bio_ClinicalBERT, jointly fine-tuned for event type and severity. The
      frozen-MiniLM version was too weak to show whether the event-type clue
      helps a stronger model.
- [ ] **Target the "quiet" serious misses.** Try features for hedged,
      potential-harm phrasing, or a second-stage check for reports that
      mention a high-risk medication or event but score low. Wait until the
      sponsor answers whether these are real or label noise.
- [ ] **Cluster drug names.** Medication fields are in the input, but semantic
      grouping of drug names hasn't been built.

### Needs sponsor input

- [ ] **Triage cutoff policy:** 0.236 (misses 2–3 serious cases) or about 0.152
      (catches all, with about 25 more false alarms per day). This is a
      staffing and risk decision.
- [ ] **Are the missed serious cases real or label noise?** Raise this at the
      Safety Officer shadowing session.
- [ ] **Narrative/label mismatch:** the same question at larger scale, for reports
      that say "no harm noted" but carry a harm grade.
- [ ] **Archetype IDs**, needed to check template overlap between train and test.
- [ ] **Is Event Type filled in at intake?** If so, the `11-buckets` 11-way
      classifier predicts a field that already exists (see end of
      `PIPELINE.md`).
