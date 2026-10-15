# Harm Severity Grader

Grades free-text patient safety incident reports on the 10-grade PSRS harm
scale (A through I) so reviewers can triage the highest-risk reports first
instead of reading every one. One model produces three outputs:

- **Letter grade** (A, B1, B2, C, D, E, F, G, H, I)
- **3-bucket label**: `none` / `some` / `serious`
- **Triage score**: a calibrated probability that the patient was harmed,
  plus a flag for priority review

This repo covers the severity/triage requirement and also holds the team's
analyst app (`analyst_app/`), which shows severity, event type, recurring
themes, similar past reports and weekly trends to a reviewer. Event-type
classification itself lives in the separate `11-buckets` repo.

New to the project? Start with [PROJECT_GUIDE.md](PROJECT_GUIDE.md): goal,
status, layout, rules and who owns what. See [PIPELINE.md](PIPELINE.md) for
diagrams and full per-experiment detail.

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
`manager_comments`, `unit_actions_taken`, `shareable_lessons`,
`HPI Designation`, `Analyst-Report Type`, and Level of Investigation
(`Level of Invet`). These are the `EXCLUDED_COLUMNS` in `data_pipeline.py`,
which asserts on every load that none of the six reach the model.

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
rolling_evaluation.py       retrains month by month (test Mar–Oct) to score 303 serious
        │                   cases instead of October's 20; a development metric
        │
predict.py                  load severity_model/ → score a CSV/parquet of new reports
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

Scripts need Python 3.11; `pip install -r requirements.txt` installs
everything (the production pipeline only needs the first block). The dataset
isn't in the repo: copy or symlink the sponsor's parquet file to
`data/LOCAL_ONLY_student_facing_candidate.parquet` (gitignored), or set
`SEVERITY_DATA_PATH` to its location.

```bash
python3 severity_model.py            # trains, evaluates, saves to severity_model/
python3 severity_model_extended.py   # extended evaluation
python3 rolling_evaluation.py        # month-by-month evaluation of the serious tier (a few minutes)
python3 prefill_extra_fields.py      # HPI Designation classifier
python3 -m pytest                    # safety-critical checks (no dataset needed)
```

Experiments and superseded models live in `experiments/` (see its
[README](experiments/README.md)). They import the pipeline modules, so run
them from the repo root as modules, e.g. `python3 -m experiments.error_analysis`,
which lists the harmed patients the saved production model fails to flag.

### Scoring new reports

```bash
python3 predict.py reports.parquet -o scored.csv   # .csv or .parquet in and out
```

`predict.py` needs a trained `severity_model/` (run `severity_model.py` once;
the `.joblib` files aren't committed). The input needs an `event_comments`
column; the intake fields (`Location Name`, `Encounter Service`, age and
medication columns) are used when present and treated as blank when absent.
Post-investigation columns such as `manager_comments` can be in the file:
they're dropped before scoring. Output has one row per input row, in order:

| Column | Meaning |
|---|---|
| `event_no` | `Event No.`, if the input has it |
| `grade` | predicted letter grade |
| `bucket` | `none` / `some` / `serious` |
| `p_none`, `p_some`, `p_serious` | bucket probabilities (sum to 1) |
| `triage_score_raw` | summed E–I probability, which the cutoff applies to |
| `triage_probability` | calibrated probability the patient was harmed |
| `flagged` | `triage_score_raw` ≥ the cutoff frozen on September |

The narrative isn't copied to the output. Scoring the raw October file this
way reproduces `evaluation.json` exactly (recall 96.0%, precision 28.9%,
exact-grade accuracy 76.6%).

### Tests

`tests/` covers the excluded-column assertion (including that leaked columns
in an input file don't change scores), `Event No.` split decoding, the
grade → bucket mapping, `bucket_probs` summing to 1 (including a model
missing a grade and shuffled class order), and a save → load → score round
trip through `predict.py`. The rolling-fold tests check that each fold
trains only on months before its cutoff and test months, no report is tested
twice, and the last fold is the production split. One extra test runs against
the real `severity_model/` when it's been trained, and is skipped otherwise.

Outputs in `severity_model/`: `evaluation.json`, `extended_evaluation.json`,
`rolling_evaluation.json` and `config.json` are committed; the `.joblib` model
files are gitignored.

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

### Serious tier, rolling evaluation (development metric)

October has only 20 G/H/I cases, so `rolling_evaluation.py` repeats the
production split month by month: train on all earlier months, pick the cutoff
on the next month, test on the month after. Testing March through October
scores **303 serious cases**, each by a model that never saw its month. The
last fold is the production split and reproduces its cutoff. Numbers are from
`severity_model/rolling_evaluation.json`.

| G/H/I cases | Flagged by triage cutoff | Labelled `serious` |
|---|---|---|
| All, Mar–Oct | 297 / 303 = 98.0% (95% CI 95.7–99.1%) | 157 / 303 = 51.8% (46.2–57.4%) |
| Grade G | 68 / 69 = 98.6% | 13 / 69 = 18.8% (11.4–29.6%) |
| Grade H | 109 / 112 = 97.3% | 76 / 112 = 67.9% (58.7–75.8%) |
| Grade I | 120 / 122 = 98.4% | 68 / 122 = 55.7% (46.9–64.2%) |
| Mar–Aug only (oversampled months) | 263 / 267 = 98.5% (96.2–99.4%) | 133 / 267 = 49.8% |
| Sep–Oct only (natural harm rate) | 34 / 36 = 94.4% (81.9–98.5%) | 24 / 36 = 66.7% |

How to read it:

- **Use it to compare models, not as a headline.** January–August informed
  every modelling choice, so October alone is still the final test.
- **The triage cutoff is close to its ceiling on this tier** (6 misses in
  303), so it can't show much improvement. The `serious` label (146 misses)
  and the share of serious cases caught at a 5% false-alarm rate
  (277 / 303 = 91.4%) leave more room.
- **The `serious` label is the weak spot, grade G most of all.** About four of
  five permanent-harm reports are flagged for review but not labelled `serious`.
- **The `serious` label needs training data.** It labels 28% of serious cases
  correctly in the March fold (trained on January alone) and 52–75% in the
  last six folds, so the pooled 51.8% understates the production model.
- **Precision and flag rate are not pooled.** January–August oversamples harm
  (about 15% of reports versus 4–5% in September and October), which inflates
  both; the per-fold values are in the JSON.
- **March–August may be slightly easier than a natural month.** 98.5% versus
  94.4% flagged, but with 36 natural-rate cases the intervals overlap and the
  gap can't be confirmed.

### Compared with the sponsor's model

The sponsor reports one number for their model: **95.2% of harmed patients
caught, while 96% of reports with no harm are correctly left alone**. To
compare fairly, we ran the sponsor's own training code on our dataset, with
our month-based split, and scored every model the same way on the same
October reports (403 harmed patients). Each model was trained once.

| Model | What it reads | Harmed patients caught |
|---|---|---|
| Our current model (word-counting, `severity_model.py`) | Report as submitted | 90.3% (364) |
| Our word-counting model, improved settings | Report as submitted | 91.6% (369) |
| Sponsor's model (Clinical-Longformer) | Report as submitted | 91.6% (369) |
| Sponsor's model, their original inputs | Report **plus** manager comments and unit actions | 93.8% (378) |
| **Our DistilBERT model** | Report as submitted | **94.3% (380)** |
| *Sponsor's reported number* | *Their real data, random split, plus manager comments and unit actions* | *95.2%* |

What this shows, in plain terms:

- **Given the same information, our DistilBERT model beats the sponsor's
  model**: 94.3% vs 91.6%, about 11 more harmed patients caught.
- **Even when their model also reads the manager comments and unit actions,
  it doesn't beat ours** (93.8% vs 94.3%, a 2-patient difference, so
  roughly a tie). Those two fields are written after the event is reviewed,
  so they partly give away the answer, and our model doesn't use them.
- **Those after-review fields are worth about 2 points** to the sponsor's
  model (91.6% → 93.8%).
- **The rest of the gap to 95.2% comes from their setup, not a better
  model.** On our data and split, their own code scores 93.8%. Their 95.2%
  was measured on real hospital data with a random split.
- **DistilBERT also raises fewer false alarms.** At the cutoff picked on
  September, it flags 10.7% of October reports and 43% of the flagged ones
  are real harm cases. Our current model flags 16.1%, with 29% real harm
  cases. Its recall at that cutoff came in at 94.8%, just under the 95%
  target, so the cutoff needs a small adjustment.
- **The sponsor's model is better at the exact letter grade** (85–86%
  correct, vs 79% for DistilBERT and 77% for our current model). Ours is
  better at the triage question: was the patient harmed?

All of this is on synthetic data, so the numbers will shift on real reports.
Details, including the combined event-type model, are in
[PIPELINE.md](PIPELINE.md). The DistilBERT results are in `distilbert_results/`
and the sponsor-model results in `experiments/sponsor_comparison/`. The
sponsor's code itself isn't in the repo.

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
  evaluated at all. The rolling evaluation above covers 303 cases (122 grade
  I) as a development metric.
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

- [x] **Add a prediction entry point.** `predict.py` loads `severity_model/`
      and scores a CSV/parquet file (see *Scoring new reports*).
      `demo_predict()` now goes through it too.
- [x] **Add tests for the safety-critical pieces.** `tests/`, run with
      `python3 -m pytest tests`.
- [x] **Separate production code from experiments.** `baseline_grader.py`,
      `ordinal_vs_coarse.py`, `combined_mtl.py`, `augment_rare_grades.py` and
      `error_analysis.py` (and the DistilBERT, leakage-check and Longformer
      experiments) are in `experiments/`; `data_pipeline.py`,
      `severity_model.py`, `severity_model_extended.py`,
      `rolling_evaluation.py` and `predict.py` are the pipeline.
- [x] **Point `error_analysis.py` at the production model.** It loads
      `severity_model/` and applies the frozen cutoff; on October it lists the
      16 of 403 harmed patients not flagged.
- [x] **Add `requirements.txt`**, with versions pinned to the environment
      the results were produced in.
- [x] **Rolling evaluation of the serious tier.** `rolling_evaluation.py`
      scores 303 G/H/I cases instead of October's 20 (see *Results*).
- [x] **Bring the analyst app into this repo.** `analyst_app/` uses this
      repo's saved severity model by default.

### Experiments

- [x] **Train DistilBERT on all 10 grades, reading longer reports.** Done at
      256 tokens (cuts off 2% of reports, vs 26.5% at 128). It catches 94.3%
      of harmed patients at 96% specificity, vs 90.3% for the current model.
      See *Compared with the sponsor's model*.
- [x] **Run the sponsor's Clinical-Longformer head-to-head.** Done on our data
      and split, twice: with their original inputs (93.8%) and with only
      the report as submitted (91.6%).
- [x] **Retry the combined model on a fine-tuned encoder.** Done with
      `experiments/combined_finetune.py`. Giving the severity model the
      predicted event type made no difference to catching harm (93.8% with
      and without it). It caught all 20 serious October cases, but so did
      the same model without the hint, and 20 cases is too few to be sure.
- [x] **Try cheap improvements to the word-counting model.**
      `experiments/improve_sensitivity.py` tested 11 settings on June–September.
      The best (letter-level features, C=4) reached 91.6% on October, vs 90.3%.
- [ ] **Decide whether DistilBERT replaces the current model.** It beats it on
      catching harm and raises fewer false alarms, but it needs a GPU to
      train, is harder to explain, and its cutoff needs adjusting on
      September (recall came in at 94.8%, just under the 95% target). If
      adopted, `predict.py` and `analyst_app/` need to load it.
- [ ] **Train DistilBERT more than once.** Each result above is one training
      run; a second or third run with a different random start would show
      how much the numbers move.
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
