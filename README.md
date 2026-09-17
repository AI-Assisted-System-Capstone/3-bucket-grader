# 3-Bucket Harm Severity Grader

Classifies free-text safety incident reports into 3 harm severity buckets, collapsed
from a 9-grade harm scale, so reviewers can triage the highest-risk reports faster
than reading all of them manually.

## Reports

The source dataset has 23 columns per report (event metadata, medication fields,
location, and four free-text narrative fields). Of those, only one column drives the
label: **Significance (PSRS Harm score)**, a 9-grade scale (A through I) collapsed
into 3 buckets:

| Bucket | Grades | Meaning |
|---|---|---|
| `none` | A, B1, B2, C, D | Unsafe condition / near miss / no harm |
| `some` | E, F | Harm, treated (incl. added hospitalization) |
| `serious` | G, H, I | Permanent harm / near death / death |

On the current 80k-row dataset (67,003 usable rows after dropping missing-harm-score
and empty-text rows):

| Bucket | Count | Share |
|---|---|---|
| `none` | 58,677 | 87.6% |
| `some` | 7,931 | 11.8% |
| `serious` | 395 | 0.6% |

## Fields

Of the 23 columns, only 3 free-text fields are currently used as model input:

- `event_comments`
- `manager_comments`
- `unit_actions_taken`

These are concatenated into a single text blob per report. (`shareable_lessons` and
the medication columns, e.g. `ME - Prescribed - Name`, `ADR - Suspect Med Name`, are
not yet used — see To-Dos.)

## Tokenize

Text is tokenized with the DistilBERT tokenizer at a max length of **128 tokens**.
This was a starting guess based on the rough sense that an average report runs
70-100 words, not yet verified against the actual token-length distribution of the
dataset — see To-Dos.

## Model

Two models exist so far:

1. **Baseline**: TF-IDF (unigrams + bigrams, top 20k features) + Logistic Regression,
   with `class_weight="balanced"` to counter the severe class imbalance.
2. **DistilBERT** (`distilbert-base-uncased`): 6 transformer layers, ~66M parameters —
   a small model, fine-tuned on the same 3-bucket task and same train/test split as
   the baseline, with class-weighted cross-entropy loss. Not yet run on the full
   dataset locally (needs `torch`/`transformers`, availability unconfirmed).

**To-do:** find a better model than DistilBERT — it's a small model and this task
needs high sensitivity; something with more capacity may be worth the added
inference cost.

## Classify

Predictions currently use **plain argmax** over the softmax output (whichever bucket
gets the highest predicted probability wins). The sponsor has asked for **high
sensitivity** (recall) as a hard requirement, particularly on `serious` and `some`,
even at the cost of precision — plain argmax doesn't let us bias toward that. Next
step is to sweep decision thresholds against the predicted probabilities and plot a
sensitivity/precision-vs-threshold curve to choose a lower cutoff deliberately,
rather than guessing one.

## Results (as of 2026-09-17, on full 80k dataset)

**Dummy baseline** (always predict `none`): 88% accuracy, 0% recall on both
minority classes — accuracy alone is meaningless here given the imbalance.

**TF-IDF + Logistic Regression**: 96% accuracy, 87% macro F1.

| Bucket | Precision | Recall |
|---|---|---|
| `none` | 0.98 | 0.97 |
| `some` | 0.79 | 0.88 |
| `serious` | 0.83 | 0.76 |

Top words pushing predictions toward `serious`: *died, permanent, urgent, death,
critical, life threatening, neurological* — semantically sensible, not spurious
correlations.

**DistilBERT**: not yet run on the full dataset locally.

## Open questions / To-Dos

- [ ] **High sensitivity is a hard requirement** (sponsor). Current `serious` recall
      is 0.76 — misses about 1 in 4 true serious cases. Needs to go up, even at the
      cost of precision.
- [ ] **Threshold vs. argmax**: build a threshold sweep (sensitivity/precision vs.
      threshold) before picking a low cutoff for high-sensitivity behavior.
- [ ] **~249 harm reports currently misclassified as `none`** — need targeted error
      analysis on these false negatives to find patterns.
- [ ] **Column scope**: currently only 3 of 23 columns are used. Sponsor wants to
      keep the medication column (drug prescribed/administered) in for **semantic
      clustering** — to check whether a department is over-ordering or
      over-prescribing a particular drug. Keep the rest lean; don't add columns
      just because they exist.
- [ ] **Label the merged text fields**: `event_comments`, `manager_comments`, and
      `unit_actions_taken` are currently concatenated with no field markers — add
      labels/prefixes so the model can tell the fields apart.
- [ ] **Time-based validation**: current train/test split is random stratified. Need
      to re-run with a month-based split (e.g. Jan-Aug train / Sep val / Oct test)
      to see whether performance holds when predicting forward in time, not just on
      a random hold-out.
- [ ] **Find a better model than DistilBERT** — current model is small (6 layers,
      66M params); worth evaluating larger/alternative architectures given the
      sensitivity requirement.
- [ ] **Verify `MAX_LENGTH=128`** against the actual token-length distribution of
      the dataset instead of the current word-count guess.
