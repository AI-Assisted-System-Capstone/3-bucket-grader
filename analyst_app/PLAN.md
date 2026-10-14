# Integration MVP plan — 2026-10-02

## Inspection completed before implementation

- Severity repository: `AI-Assisted-System-Capstone/3-bucket-grader`, commit
  `b147d62c89323c21915907a4ea7df4b4facd9553`. Read README, relevant PIPELINE
  architecture/production/comparison sections, `data_pipeline.py` and production
  `severity_model.py`. No standalone inference interface exists beyond demo_predict.
- Event-type repository: `AI-Assisted-System-Capstone/11-buckets`, commit
  `2b7d2743c5996d977fe627e7296280ca707fe9b7`. Read README and source pipeline,
  baseline checkpoint saving, and multi-task head architecture.
- Neither checkout contains trained weights. Severity includes configuration and
  evaluation JSON only. Event type includes logs/metrics only. No semantic_clustering.py
  is present. Evaluation results and training code must remain unchanged.
- Existing local CLI, requirements, scripts and tests inspected. Preserve that workflow;
  add a separate normalized integration package and web client.

## Exact inference requirements

Severity: configured upstream checkout; `severity_model/{vectorizer,model,calibrator}.joblib`
and `config.json` with `cutoff_raw_triage`. Import upstream `build_input_text`,
`bucket_probs`, `triage_score` without running training. Respect classifier.classes_
ordering, grade indices A/B1/B2/C/D/E/F/G/H/I, and calibrator positive class.
Use trusted joblib files only (pickle can execute code); match training sklearn version.

Event type: configured upstream checkout; `outputs/baseline/model_seed0.pt` and
`train_log_seed0.json`; frozen local MiniLM encoder; upstream EmbeddingMultiTask(384,
use_event_probs=True, hidden=256, dropout=.2), strict state-dict load and eval mode.
Preserve raw, unnormalized MiniLM embeddings for this trained head. Return Head A
only in upstream class order ADR/C/E/EQ/FALL/I/ME/O/SH/SI/T. Do not use hurt head.
Fine-tuned BioClinicalBERT is not trained per upstream README and is out of scope.

## Implementation sequence

1. Normalized Pydantic contracts, exact source-field adapters, inference wrappers
   with load_model/predict_event/predict_batch and explicit deterministic fixture mode.
2. Local SQLite persistence for all nine required tables, queue/audit interfaces,
   independent AI failures and startup-only model loading.
3. New MiniLM cosine vector-store interface, persisted embeddings, historical top-N,
   HDBSCAN/agglomerative themes, multi-label memberships, deterministic naming,
   explainable weekly zero-filled trend baselines. Preserve the original CLI's
   documented TF-IDF fallback. The new API has an explicit lexical hashing test
   backend for deterministic offline tests; never feed it into a trained head.
4. FastAPI routes + synthetic seed command + same-origin React/TypeScript analyst UI.
5. Synthetic contract/integration tests, original tests, frontend typecheck/build,
   local API/browser verification. Document unavailable artifacts and production gaps.

## Safety and scope

All inference and data remain local. No cloud model calls. No source reports,
embeddings, databases, weights or credentials committed. Use opaque internal UUIDs
in URLs; raw report IDs and narratives stay in bodies. No request/access logging.
Analyst decisions stay separate from AI suggestions. This is a loopback-only prototype,
not an authenticated, encrypted, hospital-ready system. No retraining or clinical claims.

## Completion record

Implemented the integration modules, standalone severity entrypoint, SQLite storage,
all required routes, synthetic seed command and analyst UI. Verified with 24 tests
including upstream contracts, frontend build/typecheck, static checks and local
desktop/mobile browser checks. See MVP_GUIDE.md and MVP_VERIFICATION.md. Real
classifier validation remains blocked by missing trained artifacts; no source
MIDAS dataset or cloud inference was used.
