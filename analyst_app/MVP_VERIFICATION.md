# Verification record

## Dashboard follow-up — 2026-10-13

- Dashboard now prioritizes a six-report review queue, with compact pattern
  context alongside it and a recent-reports switch. New-report, search, pattern
  and worklist shortcuts are functional.
- Clickable totals open global, server-filtered worklists: awaiting review,
  priority awaiting review, analysis issues and reviewed. Filters apply before
  pagination and are encoded in the URL for back/forward navigation.
- Analyst review is separate from AI status. A report is reviewed once any
  analyst decision exists; duplicate decisions do not inflate counts. This is not
  event resolution, and re-analysis does not automatically reopen the queue.
- One SQLite read snapshot supplies totals, queues and pattern metadata. The
  dashboard response excludes narratives, medications, analyst notes and vectors.
- Refresh preserves the prior snapshot while loading or after failure, clearly
  marks stale displayed data, provides retry, and times out after 15 seconds.
  Initial failure, empty workspace and completed review queue are distinct states.
- Older trend periods are labeled Past period rather than current emerging
  signals. No clustering or classifier methodology changed.

Latest verification: 16 frontend tests passed; 25 Python tests passed with three
optional upstream-source skips (28 discovered). TypeScript/production build,
`npm audit` (zero known vulnerabilities) and whitespace checks passed.

Browser checks used only a new authored-synthetic QA database under
`/tmp/midas-dashboard-qa.zvCVjA/qa.sqlite3`, served on the already permitted local
preview port 5173. Recording a decision changed awaiting/priority counts 38→37,
reviewed 0→1, removed the report from the queue and retained it in Recent reports.
Verified filtered empty states, shortcuts, 390px/320px layouts without page
overflow, and keyboard skip link. Stopping/restarting only the QA API verified
that a failed refresh retains six rows and retry recovers. No console errors in
the final reloaded QA session. QA services are stopped after verification; the
user's app remains on port 8000 with its existing database.

Scope limits: this remains a local prototype, not hospital-approved software.
The summary projection reads the workspace collections in memory and is not an
80k-report scale implementation. The trained classifier artifacts are still
missing; demo suggestions remain labeled fixtures. The user subsequently requested
publication of these source changes and handoff documents to the private repository;
local datasets, databases, generated outputs and credentials are excluded.

## UI/UX refinement — 2026-10-13

Initially implemented locally; included in the subsequent user-requested source publication.

- A consistent visual hierarchy now covers the dashboard, worklist, intake,
  report review, search, pattern list and pattern detail. Teal identifies primary
  actions; amber identifies review signals; stable states and supporting metadata
  use readable neutral colors. Color is accompanied by text labels.
- Human decisions retain the primary action treatment; AI probabilities and model
  versions are secondary disclosures. Fixture warnings remain visible. Rebuild is
  an explicitly described maintenance action, not the main task on the trends page.
- Added loading states, actionable errors, in-flight duplicate-submit protection,
  an explicit Apply action for historical result counts, and navigation directly to
  newly saved reports. Search results clear when the query/count changes. Refresh
  clears the outdated rebuild-queued message.
- Added keyboard focus rings, a working skip link, active worklist navigation on
  report detail, accessible table regions, 44px primary controls, reduced-motion
  support and narrow-screen layouts.

Verification performed:

| Check | Result |
|---|---|
| `./run_tests.sh` | 21 passed; 3 optional upstream-source tests skipped (24 discovered) |
| `cd frontend && npm test` | 8 new UI regression tests passed, using authored fixtures and mocked HTTP only |
| `npm run build` | TypeScript check and production Vite build passed |
| `npm audit` | Zero known vulnerabilities across runtime and development dependencies at this check |
| `git diff --check` | Passed |
| Browser: report intake | Required fields block invalid save; valid synthetic report opens its detail page |
| Browser: duplicate source ID | Clear error; existing draft preserved; no duplicate saved |
| Browser: search and historical matching | 12 results returned for both workflows using real local MiniLM |
| Browser: analyst review | Explicit B2 / ME decision saved separately; AI fixture grade remained E |
| Browser: patterns | Rebuild completed, cards/detail/representative links loaded; stale queued notice cleared on refresh |
| Browser: keyboard | Skip link moved focus to main content |
| Browser: responsive | Desktop checked at 1280px; 390px mobile screens inspected; 320px dashboard/report detail had no page overflow; worklist table scrolls within its region |
| Browser console | No errors/warnings in the final checked session |

Browser writes used a new, isolated synthetic QA database under
`/tmp/midas-ux-qa.ZKVALa/qa.sqlite3` (38 seeded examples + one UI test report).
The existing `local_state/demo.sqlite3` and all hospital input/output files were
untouched. The QA server was loopback-only and is stopped after testing. Temporary
QA paths are not durable project dependencies. No cloud model calls or deployment.

The frontend now has a separate bootstrap entrypoint and a Vitest/Testing Library
suite. Install lockfile dependencies with `npm ci` before running `npm test`.
Vitest 4 requires Node 20, 22, or 24+; this run used Node 26.5.0. The existing macOS
Python environment still prints its LibreSSL/urllib3 warning. This work does not
validate real classifier accuracy, clinical usability, production readiness or
full accessibility conformance; the original classifier weights are still missing.

## Integration release — 2026-10-02

## Result

- **24 tests passed, no skips**, when both inspected upstream checkout paths were
  supplied. Includes the three pre-existing CLI pattern tests.
- Plain `./run_tests.sh` also passed: 21 executed tests passed and the three
  optional upstream-source tests were skipped without their environment paths.
- TypeScript type checking and production Vite build passed.
- Ruff checks passed for the new Python integration modules/tests.
- `npm audit --omit=dev` reported zero known runtime dependency vulnerabilities.
- Browser verification passed for dashboard, event review, search, pattern list,
  pattern detail, adjustable result counts and analyst override. No browser errors
  were reported in the checked session. Mobile viewport 390×844 had document width
  390 (no horizontal page overflow); desktop checked at 1440×1000.
- Existing training and evaluation files were not changed. The standalone severity
  inference entrypoint was added only to the local inspection checkout; a durable
  copy is in integrations/severity_inference.py. Nothing was pushed upstream.

## What is real versus synthetic

Real cached local `sentence-transformers/all-MiniLM-L6-v2` generated all demo
embeddings. All 38 vectors are 384-dimensional, finite and unit-normalized. The
cosine index reloads from local storage after restart. HDBSCAN discovered three
themes, containing 16, 11 and 11 events at the configured .80 membership threshold.
Weekly current/baseline counts were 8/1, 3/1 and 3/1.

Severity and event type are **explicit deterministic fixtures** because neither
upstream repository includes the trained artifacts. The yellow UI banner and
per-result fixture flags disclose this. All demo reports were authored for this
implementation; no source MIDAS reports were used. Fixtures and temporary
zero-weight/test-class artifacts validate interfaces, never model accuracy.

The inspected demo database contained:

| Table | Rows |
|---|---:|
| events | 38 |
| ai_results | 38 |
| event_embeddings | 38 |
| similar_events | 38 |
| themes | 3 |
| event_themes | 38 |
| trend_results | 3 |
| analyst_feedback | 1 |
| audit_log | 79 |

The browser-saved analyst decision was B2 while the original fixture AI suggestion
remained E. Tests also verify that analyst notes/narratives/source IDs are absent
from audit payloads, and that exceptions/validation responses do not echo input.

## Demo query

Query: “A medication was nearly administered at the wrong dose after a barcode
scanner problem.”

| Matching source event | Cosine similarity |
|---|---:|
| DEMO-BARCODE-06-00 | 0.686547 |
| DEMO-BARCODE-04-00 | 0.686547 |
| DEMO-BARCODE-08-01 | 0.686547 |
| DEMO-BARCODE-08-07 | 0.686547 |
| DEMO-BARCODE-08-02 | 0.686547 |

These ties are expected: demo barcode cases repeat an authored narrative at
different dates. All are associated with the medication/barcode theme in Medical
unit, Demo campus. Tie order uses internal IDs, so a newly seeded database may
order equal-score reports differently. This is not a retrieval-quality benchmark.
The browser search also returned eight results when requested, and an event detail
returned twelve strictly earlier reports. The original CLI's novelty output is
separate and was not rerun against the private source dataset in this integration.

## Commands executed

Inspection (read-only GitHub source retrieval; reproducible checkout commands):

```bash
gh repo view AI-Assisted-System-Capstone/3-bucket-grader
git clone --depth 1 https://github.com/AI-Assisted-System-Capstone/3-bucket-grader /tmp/midas-upstream.d9JNJj/severity
git clone --depth 1 https://github.com/AI-Assisted-System-Capstone/11-buckets /tmp/midas-upstream.d9JNJj/event_type
git -C /tmp/midas-upstream.d9JNJj/severity rev-parse HEAD
git -C /tmp/midas-upstream.d9JNJj/event_type rev-parse HEAD
```

Setup/build/test (existing Python virtual environment reused):

```bash
.venv/bin/python -m pip install 'fastapi>=0.115,<0.120' 'uvicorn>=0.30,<0.36' 'httpx>=0.27,<0.29' 'pytest>=8,<9'
cd frontend
npm install --no-audit --no-fund
npm run typecheck
npm run build
npm audit --omit=dev --json
cd ..
uvx ruff format backend integrations tests/test_integration.py
uvx ruff check backend integrations tests/test_integration.py
MIDAS_TEST_SEVERITY_REPO=/tmp/midas-upstream.d9JNJj/severity MIDAS_TEST_EVENT_TYPE_REPO=/tmp/midas-upstream.d9JNJj/event_type ./run_tests.sh
git diff --check
```

Demo:

```bash
MIDAS_DATABASE=local_state/demo.sqlite3 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 .venv/bin/python -m backend.seed --fixture-mode
MIDAS_DATABASE=local_state/demo.sqlite3 MIDAS_FIXTURE_MODE=true ./run_app.sh
```

Browser checks used `npm exec --yes --package=agent-browser -- agent-browser`
with a dedicated local session: open, snapshot, fill, select, click, errors,
viewport and screenshot commands. An additional local-only HTTP/SQLite assertion
check verified live schemas, search, embedding dimensions/normalization, table
counts, trends, and preserved analyst-versus-AI decisions. No report-bearing
requests went to external services. React review prompted deliberate required
analyst selection (no default A/ADR decision); browser checks verified the visible
fixture disclosure, navigation and responsive layout.

The first test attempt caught a multiline-context-manager syntax error, which was
fixed. A browser retry was needed while the local MiniLM was still starting. The
final suite/build/browser checks above passed. The existing LibreSSL/urllib3
environment warning remains; inference here is offline.

## Limitations

Real classifier predictions cannot be verified until teammates supply the original
trusted artifacts and compatible dependencies. No MIDAS connector, authenticated
analyst identity, hospital access control or production deployment is implemented.
Theme defaults require reviewer validation. See MVP_GUIDE.md for the complete
environment matrix, file list, architecture, data-locality controls and scale limits.

## Pre-push end-to-end verification

The original 38-report demo above was retained. A 39th authored synthetic report,
`DEMO-PUBLISH-E2E-20261002`, was submitted through the browser worklist form.

| Boundary | Result | Evidence |
|---|---|---|
| Browser → API | Pass | Report submitted and event-detail page rendered |
| API → SQLite | Pass | New event and its 384-dimensional MiniLM embedding persisted |
| Independent AI services → response | Pass in fixture mode | Unified result complete; both classifier fixtures disclosed |
| Similarity → UI | Pass | Top N changed from 5 to 12; twelve ranked, strictly earlier cases returned |
| Membership → themes/trends | Pass | New report joined medication theme; theme count increased from 16 to 17 |
| Database → dashboard | Pass | Event count increased from 38 to 39 |
| Analyst decision → storage | Pass | Browser selected B2/ME independently of fixture AI grade E |
| Restart → recovered state | Pass | Added regression test checks events, suggestions, decisions, vector index, themes and trends after a fresh app startup |

`npm ci`, TypeScript checking, Vite production build, Ruff, `pip check` and
`git diff --check` passed again. Browser error checks reported no errors. Native
date-field automation required setting its ISO date value before submitting; the
form correctly rejected the initially empty date. No application fix was needed.

Publication targets the existing private `AI-Assisted-System-Capstone/midas-pattern-phase-local`
repository. Only source code, authored test fixtures, dependency manifests and
documentation are included; local report files, databases, embeddings, trained
artifacts, build output, environments and credentials are excluded. Real trained
classifier validation remains blocked by the missing original artifacts.
