# 3-Bucket Harm Severity Grader

Classifies free-text safety incident reports into 3 harm severity buckets, collapsed
from a 9-grade harm scale, so reviewers can triage the highest-risk reports faster
than reading all of them manually.

See [PIPELINE.md](PIPELINE.md) for how this pipeline works end to end (with
diagrams), how it relates to the teammate's event-type pipeline, and the full
results for every model variant tried so far.

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
and empty-text rows), split by the dataset's real Jan-Aug/Sep/Oct date split
(see "Fields" and "Results" below):

| Bucket | Train (Jan-Aug) | Val (Sep) | Test (Oct) |
|---|---|---|---|
| `none` | 42,523 (84.9%) | 8,245 (95.9%) | 7,909 (95.2%) |
| `some` | 7,209 (14.4%) | 339 (3.9%) | 383 (4.6%) |
| `serious` | 359 (0.7%) | 16 (0.2%) | 20 (0.2%) |

Note the training set's harm rate is deliberately higher than validation/test
(the synthetic data over-samples harm cases in training, same as the teammate's
11-buckets dataset) — validation and test reflect the natural rate.

## Fields

**Fixed 2026-09-22:** the model input used to include `manager_comments` and
`unit_actions_taken`, both filled in *after* a safety officer investigates a
report. A model reading them is scoring hindsight, not predicting from what's
available at submission time — this is the same leakage our teammate's
11-buckets project flags and hard-blocks in its pipeline. It's also almost
certainly why this project's earlier numbers (97% acc, 0.82 `serious` recall)
looked stronger than their event-type/hurt baseline (PR-AUC 0.47): different,
easier task, not a better model.

Model input is now built by `data_pipeline.py`, shared by both `baseline_grader.py`
and `finetune_distilbert.py`, and matches the teammate's `src/pipeline.py`
feature set so both projects can be compared on the same rows:

- `event_comments` (the narrative — the only free-text field still used)
- a short intake-field prefix, available at submission time: unit
  (`Location Name`), service (`Encounter Service`), age (`Age at Encounter`),
  and prescribed/administered/suspect medication + dose

`manager_comments`, `unit_actions_taken`, and `shareable_lessons` are never
read from disk. `data_pipeline.py` asserts this on every load.

## Tokenize

Text is tokenized with the DistilBERT tokenizer at a max length of **128 tokens**.
This was a starting guess based on the rough sense that an average report runs
70-100 words, not yet verified against the actual token-length distribution of the
dataset — see To-Dos.

## Model

**Production model, as of 2026-09-24: `severity_model.py`.** TF-IDF (unigrams +
bigrams, top 20k features) + Logistic Regression, trained on the full 10-grade
PSRS scale (not just 3 buckets), `class_weight="balanced"` to counter the
severe class imbalance. Full architecture, code, and how the letter grade,
3-bucket label, and triage score are all derived from this one model:
see `PIPELINE.md`.

`baseline_grader.py` (3-bucket only) and `finetune_distilbert.py` (DistilBERT,
also 3-bucket only) are earlier, superseded versions — kept for history, not
run going forward. DistilBERT remains a candidate to swap in for TF-IDF
*inside* `severity_model.py` if it's re-run on the 10-grade label and shown to
beat these numbers; it hasn't been re-run since the leakage fix, so there's no
current evidence either way.

## Classify

**Done, 2026-09-24.** The triage cutoff is no longer plain argmax — it's
picked on the validation set for the sponsor's stated ≥95% recall requirement,
frozen, and applied unchanged to test (96.8% recall achieved there, at 27.5%
precision — flagging about 1 in 6 reports). The raw score is also calibrated
(Platt scaling) so it reads as an honest probability, not just a ranking
signal, correcting for the training set's oversampled harm rate. Full numbers
in `PIPELINE.md` under "Final, saved model."

## Results

### Current (as of 2026-09-22): leakage-safe fields, real time-based split (Oct test set)

Train = Jan-Aug (50,091 rows), validation = Sep (8,600), test = Oct (8,312) —
the dataset's real time split, same rows the teammate's 11-buckets model uses,
via `data_pipeline.py`.

**Dummy baseline** (always predict `none`): 95% accuracy, 0% recall on both
minority classes — accuracy alone is meaningless here given the imbalance.

**TF-IDF + Logistic Regression**, test set (Oct): 96% accuracy, 77% macro F1.

| Bucket | Precision | Recall |
|---|---|---|
| `none` | 0.99 | 0.96 |
| `some` | 0.55 | 0.89 |
| `serious` | 0.55 | 0.80 |

`serious` recall (0.80) held up well even after dropping the leaky fields —
the words that drive it (*died, permanent, deceased, death, life threatening*)
live in the narrative itself, not in the post-investigation comments. Precision
dropped (0.83 → 0.55 on `serious`), which is the honest cost of removing
hindsight the model previously had access to.

**DistilBERT**: not yet re-run on the fixed pipeline — `finetune_distilbert.py`
is updated to use `data_pipeline.py` but takes ~30 min on this machine; run it
next and update this table.

### Combined multi-task experiment (2026-09-24) — tried, did not beat the baseline

`combined_mtl.py` tests the merge idea head-on: does adding the teammate's
11-bucket event type as an input clue improve severity prediction? Architecture
mirrors their `src/mtl.py` (frozen `all-MiniLM-L6-v2` encoder, Head A = 11-way
event type, Head C = 3-way severity fed Head A's softmax probs), trained
jointly, 3 seeds, with a no-clue ablation for comparison, on the same
leakage-safe / date-split data as everything else here.

| Metric | With event-type clue | Without clue (ablation) |
|---|---|---|
| Event-type accuracy | 0.831 ± 0.006 | 0.831 ± 0.006 |
| `serious` precision | 0.087 ± 0.017 | 0.073 ± 0.007 |
| `serious` recall | 0.517 ± 0.062 | 0.583 ± 0.103 |
| `some` precision | 0.190 ± 0.020 | 0.189 ± 0.009 |
| `some` recall | 0.767 ± 0.038 | 0.762 ± 0.017 |
| macro F1 | 0.451 ± 0.020 | 0.444 ± 0.005 |

(Event-type accuracy 83.1% is a sanity check that the setup is correct — close
to the teammate's own reported 87.3% on the same architecture/encoder.)

**Two findings, both negative for this specific approach:**

1. **The event-type clue does not measurably help severity prediction.** The
   with/without-clue gap (0.451 vs 0.444 macro F1) is smaller than the
   seed-to-seed noise (±0.02). This replicates the teammate's own finding on
   their frozen-encoder hurt/not-hurt task ("a frozen encoder may simply not
   be able to make use of the clue") -- independently, on a different task.
2. **The MiniLM multi-task setup underperforms the plain TF-IDF baseline by a
   lot**: macro F1 0.45 here vs. 0.77 for TF-IDF + LogReg above, with
   `serious` precision falling from 0.55 to 0.07-0.09. A frozen, generic
   384-dim sentence embedding loses the sharp, low-frequency words ("died,"
   "permanent," "deceased") that a 20k-feature TF-IDF vector captures
   directly and that drive `serious` detection.

**Conclusion: do not ship this.** The "cheap" version of the merge (their
frozen-encoder architecture, borrowed as-is) costs more than it gives back.
If the event-type-as-clue idea is worth testing again, it needs to sit on top
of a fine-tuned encoder (DistilBERT or Bio_ClinicalBERT) that already beats
TF-IDF, not a frozen general-purpose one -- a real next step, not a re-run of
this one. Code: `combined_mtl.py`. Raw per-seed results:
`combined_mtl_results.json`.

### Previous (2026-09-17) — retired, do not cite

The numbers below used `manager_comments`/`unit_actions_taken` (post-investigation
leakage) and a random stratified split instead of the real time-based one. Kept
here only as a record of what changed, not as a result to compare against.

| Model | Accuracy | `serious` precision | `serious` recall |
|---|---|---|---|
| TF-IDF + LogReg | 96% | 0.83 | 0.76 |
| DistilBERT | 97% | 0.87 | 0.82 |

## Open questions / To-Dos

- [ ] **High sensitivity is a hard requirement** (sponsor). Current best on the
      leakage-safe pipeline (TF-IDF + LogReg) `serious` recall is 0.80 (16/20 test
      cases caught). Needs to go up further, even at the cost of precision — and
      needs re-checking once DistilBERT is re-run on the fixed pipeline.
- [x] ~~**~249 harm reports currently misclassified as `none`**~~ — done. Found 242
      false negatives (236 `some`, 6 `serious`) in the test set. Key finding: many
      of these narratives explicitly downplay harm ("no harm noted", "no adverse
      events reported") despite the official harm grade saying otherwise — a
      possible narrative/label mismatch worth raising with the sponsor. Not a
      short-text problem (FNs are *longer* on average). Mostly confident misses,
      not borderline (only 15% are "near misses" a lower threshold would fix).
- [x] ~~**Column scope**~~ — done, 2026-09-22. Medication fields (prescribed /
      administered / suspect-drug + dose) are now in the input prefix, alongside
      unit/service/age. Semantic clustering on drug names is still a separate,
      not-yet-built analysis — this just makes the fields available to the model.
- [x] ~~**Label the merged text fields**~~ — done. Text fields are now prefixed
      (`Event Comments: ...`, etc.) instead of blindly concatenated.
- [x] ~~**Drop leaked post-investigation fields**~~ — done, 2026-09-22.
      `manager_comments`/`unit_actions_taken` removed from model input; see
      "Fields" and "Results" above. `data_pipeline.py` now hard-asserts they
      never reach the model.
- [x] ~~**Time-based validation**~~ — done, 2026-09-22. `data_pipeline.py` now
      uses the dataset's real Jan-Aug/Sep/Oct split (via the `Event No.` prefix),
      matching the teammate's 11-buckets pipeline. Split sizes (50,091/8,600/8,312)
      match theirs exactly, confirming both projects are now reading the same rows.
- [x] ~~**Move from 3-bucket to the full 10-grade PSRS scale**~~ — done,
      2026-09-24. `severity_model.py` is the finished model: trains on all 10
      grades, derives the letter grade, the 3-bucket label, and the triage
      score from one model. See `PIPELINE.md` for the full writeup.
      `baseline_grader.py` is superseded, kept for history only.
- [x] ~~**Threshold vs. argmax**~~ — done, 2026-09-24, using the teammate's
      approach as the template: cutoff picked on validation for ≥95% recall
      (the sponsor's stated requirement), frozen, applied to test (96.8%
      recall achieved, 27.5% precision — the honest cost of that requirement).
      Platt-scaled calibration added on top since training data oversamples
      harm relative to validation/test. Details and numbers in `PIPELINE.md`.
- [ ] **Re-run DistilBERT** on the fixed pipeline, now as a candidate to
      replace TF-IDF inside `severity_model.py` (same threshold/calibration
      wrapper either way) — not yet done, ~30 min run. TF-IDF is the current
      production choice because it's already proven and fast; DistilBERT is
      only worth swapping in if it clearly beats these numbers, not by
      default. `finetune_distilbert.py` needs updating to train on
      `full_label` (10-grade) instead of the retired 3-bucket `label` first.
- [ ] **Verify `MAX_LENGTH=128`** against the actual token-length distribution of
      the dataset instead of the current word-count guess.
- [x] ~~**Combine with teammate's 11-buckets model (frozen-MiniLM version)**~~ —
      tried, 2026-09-24, did not beat the baseline. See "Combined multi-task
      experiment" above: event-type clue doesn't measurably help severity
      (gap smaller than seed noise), and the MiniLM multi-task setup itself
      underperforms plain TF-IDF by a lot (macro F1 0.45 vs 0.77). Not shipped.
- [ ] **Retry the combined model on a fine-tuned encoder** instead of frozen
      MiniLM (DistilBERT or Bio_ClinicalBERT, jointly fine-tuned for both
      event-type and severity). The frozen-encoder version above was cheap to
      test but too weak on its own to tell whether the event-type clue would
      help a stronger model -- that question is still open.
