import React, { useEffect, useState } from "react";
import {
  Link,
  NavLink,
  Route,
  Routes,
  useParams,
  useLocation,
  useNavigate,
  useSearchParams,
} from "react-router-dom";
import {
  Activity,
  AlertTriangle,
  ArrowRight,
  ArrowUpRight,
  BarChart3,
  CheckCircle2,
  ChevronRight,
  ClipboardList,
  Layers3,
  LayoutDashboard,
  LockKeyhole,
  Plus,
  RefreshCw,
  Search,
  ShieldCheck,
  LoaderCircle,
  X,
} from "lucide-react";
import type { AI, Event, Health, Match, Pattern, Trend } from "./types";
import { Dashboard } from "./Dashboard";
import "./styles.css";

async function api<T>(path: string, body?: unknown): Promise<T> {
  const response = await fetch(
    "/api" + path,
    body === undefined
      ? undefined
      : {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        },
  );
  if (!response.ok) {
    // Only display approved, non-sensitive server messages, never echoed inputs.
    const messages: Record<number, string> = {
      404: "This item could not be found. Return to the list and refresh.",
      422: "Check the required fields and allowed values, then try again.",
      503: "The local embedding model is unavailable. Check model status before retrying.",
    };
    const conflict =
      path === "/events"
        ? "That source event ID already exists. Use a unique ID or open the existing report."
        : path.includes("/similar")
          ? "This report's embedding is not ready. Queue analysis, then refresh."
          : "This operation is already queued. Refresh to check its progress.";
    throw new Error(
      response.status === 409
        ? conflict
        : messages[response.status] ||
          `Request failed (${response.status}). Please retry or check the local API.`,
    );
  }
  return response.json() as Promise<T>;
}
function useLoad<T>(path: string, tick = 0) {
  const [data, setData] = useState<T | null>(null),
    [error, setError] = useState("");
  useEffect(() => {
    let live = true;
    setData(null);
    setError("");
    api<T>(path)
      .then((v) => {
        if (live) setData(v);
      })
      .catch((e) => {
        if (live) setError(e.message);
      });
    return () => {
      live = false;
    };
  }, [path, tick]);
  return { data, error, loading: data === null && !error };
}
function Loading({ text = "Loading…" }: { text?: string }) {
  return (
    <div className="loading" role="status">
      <LoaderCircle size={18} className="spin" />
      {text}
    </div>
  );
}
function RouteFocus() {
  const { pathname } = useLocation();
  useEffect(() => {
    window.scrollTo(0, 0);
    document.getElementById("main-content")?.focus({ preventScroll: true });
  }, [pathname]);
  return null;
}
const patternOrder = (a: Pattern, b: Pattern) => {
  const rank = { emerging: 0, watch: 1, stable: 2 };
  return (
    (a.trend ? rank[a.trend.status] : 3) -
      (b.trend ? rank[b.trend.status] : 3) || b.event_count - a.event_count
  );
};
const pct = (n: number) => `${(n * 100).toFixed(1)}%`;
const change = (t: Trend) =>
  t.percent_change === null
    ? t.current_count
      ? "New from zero"
      : "No baseline activity"
    : `${t.percent_change >= 0 ? "+" : ""}${t.percent_change.toFixed(0)}%`;
function ErrorNote({ text }: { text: string }) {
  return text ? (
    <div role="alert" className="error">
      {text}
    </div>
  ) : null;
}
function Tag({
  children,
  kind = "neutral",
}: {
  children: React.ReactNode;
  kind?: string;
}) {
  return <span className={"tag " + kind}>{children}</span>;
}
function Heading({
  eyebrow,
  title,
  description,
  action,
}: {
  eyebrow: string;
  title: string;
  description: string;
  action?: React.ReactNode;
}) {
  return (
    <div className="heading">
      <div>
        <div className="eyebrow">{eyebrow}</div>
        <h1>{title}</h1>
        <p>{description}</p>
      </div>
      {action}
    </div>
  );
}
function Empty({ text }: { text: string }) {
  return (
    <div className="empty">
      <Layers3 size={30} />
      <p>{text}</p>
    </div>
  );
}
function Spark({ trend, large = false }: { trend: Trend; large?: boolean }) {
  const max = Math.max(...trend.timeline.map((p) => p.count), 1);
  return (
    <div
      className={"spark " + (large ? "large" : "")}
      role="img"
      aria-label={`Weekly event counts: ${trend.timeline.map((p) => p.week + ": " + p.count).join(", ")}`}
    >
      {trend.timeline.map((p, i) => (
        <div className="bar-wrap" key={p.week}>
          <div
            title={`${p.week}: ${p.count}`}
            className={i === trend.timeline.length - 1 ? "bar current" : "bar"}
            style={{ height: `${Math.max((p.count / max) * 100, 3)}%` }}
          />
          {large && <small>{p.week.slice(5)}</small>}
        </div>
      ))}
    </div>
  );
}
function PatternCard({ pattern }: { pattern: Pattern }) {
  const t = pattern.trend;
  return (
    <Link
      to={"/trends/" + pattern.theme_id}
      className={"pattern-card " + (t?.status || "")}
    >
      <div className="row">
        <span className="pattern-icon">
          <Layers3 size={18} />
        </span>
        <Tag kind={t?.status}>
          {t?.status === "emerging"
            ? "Emerging pattern"
            : t?.status || "Awaiting baseline"}
        </Tag>
      </div>
      <h3>{pattern.name}</h3>
      <p>
        {Object.keys(pattern.location_distribution).join(" · ") ||
          "No assigned locations"}
      </p>
      {t && (
        <>
          <Spark trend={t} />
          <div className="pattern-stats">
            <span>
              <b>{t.current_count}</b>
              <small>Last complete week</small>
            </span>
            <span>
              <b>{t.baseline_count.toFixed(1)}</b>
              <small>Weekly baseline</small>
            </span>
            <span className="growth">{change(t)}</span>
          </div>
        </>
      )}
      <div className="card-footer">
        Explore associated cases <ArrowUpRight size={16} />
      </div>
    </Link>
  );
}
function EventTable({
  events,
  loading = false,
  error = "",
}: {
  events: Event[];
  loading?: boolean;
  error?: string;
}) {
  if (loading) return <Loading text="Loading reports…" />;
  if (error) return null;
  return (
    <div
      className="table-wrap"
      role="region"
      aria-label="Safety reports table"
      tabIndex={0}
    >
      <table>
        <caption className="sr-only">
          Safety reports and AI suggestions. Open a report to record an analyst
          decision.
        </caption>
        <thead>
          <tr>
            <th>Event</th>
            <th>Date / location</th>
            <th>AI event type</th>
            <th>AI severity</th>
            <th>AI review signal</th>
            <th>Analyst review</th>
          </tr>
        </thead>
        <tbody>
          {events.map((e) => (
            <tr
              key={e.id}
              className={e.ai?.severity?.priority_flag ? "priority-row" : ""}
            >
              <td>
                <Link className="event-link" to={"/events/" + e.id}>
                  {e.event_id}
                  <ChevronRight size={14} />
                </Link>
                <span className="truncate">{e.event_comments}</span>
              </td>
              <td>
                {e.event_date}
                <small>{e.location_name || "Not supplied"}</small>
              </td>
              <td>
                {e.ai?.event_type?.event_type_label || "Unavailable / pending"}
              </td>
              <td>
                {e.ai?.severity ? (
                  <Tag>
                    Grade {e.ai.severity.grade} · {e.ai.severity.bucket}
                  </Tag>
                ) : (
                  "—"
                )}
              </td>
              <td>
                {e.ai?.severity?.priority_flag ? (
                  <Tag kind="watch">
                    <AlertTriangle size={13} />
                    {e.ai.severity.fixture
                      ? "Fixture priority"
                      : "Priority suggested"}
                  </Tag>
                ) : (
                  <Tag>
                    {e.ai?.status === "complete"
                      ? "No priority flag"
                      : e.ai?.status || "Unanalyzed"}
                  </Tag>
                )}
              </td>
              <td>
                <span className="review-label">
                  {e.review_status === "reviewed" ? (
                    <>
                      <CheckCircle2 size={14} />
                      Reviewed
                    </>
                  ) : (
                    "Awaiting review"
                  )}
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {!events.length && (
        <Empty text="No reports yet. Add a synthetic report to begin." />
      )}
    </div>
  );
}
function Worklist() {
  const navigate = useNavigate();
  const [params, setParams] = useSearchParams();
  const filters: Record<string, string> = {
    all: "All reports",
    unreviewed: "Awaiting review",
    priority: "Priority · awaiting review",
    reviewed: "Reviewed",
    analysis_issues: "Analysis issues",
  };
  const filter = Object.hasOwn(filters, params.get("filter") || "")
    ? params.get("filter")!
    : "all";
  const page = Math.floor(
    Math.max(1, Math.min(1000000, Number(params.get("page")) || 1)),
  );
  const offset = (Math.floor(page) - 1) * 50;
  function paginate(next: number) {
    setParams({ filter, page: String(next) });
  }
  const [tick, setTick] = useState(0),
    [show, setShow] = useState(params.get("new") === "1"),
    [error, setError] = useState(""),
    [busy, setBusy] = useState(false);
  const events = useLoad<{ items: Event[]; total: number }>(
    `/events?offset=${offset}&limit=50&filter=${filter}`,
    tick,
  );
  async function submit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    if (busy) return;
    setBusy(true);
    setError("");
    const f = new FormData(e.currentTarget);
    try {
      const saved = await api<Event>("/events", Object.fromEntries(f));
      navigate("/events/" + saved.id);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  return (
    <>
      <Heading
        eyebrow="ANALYST WORKSPACE"
        title="Safety event worklist"
        description="Review the report, consider the suggestions, record your decision."
        action={
          <button
            className={show ? "secondary" : ""}
            disabled={busy}
            aria-expanded={show}
            aria-controls="new-report"
            onClick={() => {
              setShow(!show);
              setError("");
            }}
          >
            {show ? <X size={16} /> : <Plus size={16} />}
            {show ? "Cancel new report" : "New report"}
          </button>
        }
      />
      <ErrorNote text={error || events.error} />
      <div className="worklist-toolbar">
        <label className="inline">
          Show reports
          <select
            value={filter}
            onChange={(e) => setParams({ filter: e.target.value })}
          >
            {Object.entries(filters).map(([value, label]) => (
              <option key={value} value={value}>
                {label}
              </option>
            ))}
          </select>
        </label>
        <span className="muted">
          Filters apply to the entire workspace, not just this page.
        </span>
      </div>
      {show && (
        <form
          id="new-report"
          className="panel form report-form"
          onSubmit={submit}
          aria-busy={busy}
        >
          <h2>Submit a synthetic report</h2>
          <p className="muted">
            Prototype workspace. Use authored synthetic examples only. Fields
            marked * are required.
          </p>
          <div className="form-grid">
            <label>
              Source event ID *
              <input name="event_id" required maxLength={100} autoFocus />
            </label>
            <label>
              Event date *
              <input type="date" name="event_date" required />
            </label>
            <label>
              Location
              <input name="location_name" />
            </label>
            <label>
              Facility
              <input name="facility" />
            </label>
          </div>
          <label>
            Report narrative *
            <textarea
              name="event_comments"
              required
              minLength={10}
              maxLength={30000}
            />
          </label>
          <div className="form-actions">
            <button disabled={busy}>
              {busy ? "Saving report…" : "Save report & queue analysis"}
            </button>
          </div>
        </form>
      )}
      {events.data && !events.data.items.length && events.data.total > 0 ? (
        <div className="empty">
          <h2>No reports on this page</h2>
          <p>The list may have changed since you last opened it.</p>
          <button className="secondary" onClick={() => paginate(1)}>
            Return to first page
          </button>
        </div>
      ) : events.data?.total === 0 && filter !== "all" ? (
        <div className="empty">
          <h2>No reports in this view</h2>
          <p>Try another filter or return to all reports.</p>
          <button
            className="secondary"
            onClick={() => setParams({ filter: "all" })}
          >
            Show all reports
          </button>
        </div>
      ) : (
        <EventTable
          events={events.data?.items || []}
          loading={events.loading}
          error={events.error}
        />
      )}
      <div className="pagination">
        <span>
          {events.data?.total ?? "—"} reports · showing{" "}
          {events.data?.items.length ? offset + 1 : 0}–
          {events.data?.items.length ? offset + events.data.items.length : 0}
        </span>
        <button
          className="secondary"
          disabled={events.loading || offset === 0}
          onClick={() => paginate(Math.max(1, page - 1))}
        >
          Previous
        </button>
        <button
          className="secondary"
          disabled={events.loading || offset + 50 >= (events.data?.total || 0)}
          onClick={() => paginate(page + 1)}
        >
          Next
        </button>
        <button className="secondary" onClick={() => setTick(tick + 1)}>
          <RefreshCw size={14} />
          Refresh
        </button>
      </div>
    </>
  );
}
function Matches({
  matches,
  historical = true,
}: {
  matches: Match[];
  historical?: boolean;
}) {
  return (
    <div className="matches">
      {matches.map((m, i) => (
        <Link key={m.event_id} to={"/events/" + m.event_id}>
          <span className="rank">{i + 1}</span>
          <div>
            <b>{m.source_event_id || "Matching report"}</b>
            <small>
              {m.event_date} · {m.location_name}
            </small>
          </div>
          <div className="similarity">
            {m.similarity.toFixed(3)}
            <small>Similarity score</small>
          </div>
          <ChevronRight size={15} />
        </Link>
      ))}
      {!matches.length && (
        <p className="muted">
          {historical
            ? "No eligible historical reports. Only earlier event dates are included."
            : "No matching reports in this workspace. Add and analyze reports, then search again."}
        </p>
      )}
    </div>
  );
}
const grades = ["A", "B1", "B2", "C", "D", "E", "F", "G", "H", "I"],
  codes = ["ADR", "C", "E", "EQ", "FALL", "I", "ME", "O", "SH", "SI", "T"];
function EventDetail() {
  const { id } = useParams();
  const [tick, setTick] = useState(0),
    [n, setN] = useState(5),
    [requestedN, setRequestedN] = useState("5"),
    [saving, setSaving] = useState(false),
    [analyzing, setAnalyzing] = useState(false),
    [error, setError] = useState(""),
    [notice, setNotice] = useState("");
  const report = useLoad<Event>("/events/" + id, tick),
    ai = useLoad<AI>("/events/" + id + "/ai", tick),
    matches = useLoad<Match[]>(`/events/${id}/similar?top_n=${n}`, tick);
  const r = report.data,
    a = ai.data;
  useEffect(() => {
    setNotice("");
    setError("");
  }, [id]);
  async function feedback(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    if (saving) return;
    setSaving(true);
    setError("");
    setNotice("");
    const f = new FormData(e.currentTarget);
    try {
      await api(`/events/${id}/feedback`, Object.fromEntries(f));
      setNotice("Analyst decision saved separately from AI suggestions.");
      setTick(tick + 1);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setSaving(false);
    }
  }
  async function analyze() {
    if (analyzing) return;
    setAnalyzing(true);
    setError("");
    setNotice("");
    try {
      await api("/ai/analyze", { event_ids: [id] });
      setNotice("Analysis queued. Refresh to see the result.");
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setAnalyzing(false);
    }
  }
  return (
    <>
      <Link className="back" to="/worklist">
        ← Back to worklist
      </Link>
      <Heading
        eyebrow="EVENT REVIEW"
        title={
          r?.event_id ||
          (report.error ? "Report unavailable" : "Loading report…")
        }
        description="Read the original report alongside independent AI suggestions."
        action={
          <div className="row">
            <button
              className="secondary"
              disabled={saving || analyzing}
              onClick={() => setTick(tick + 1)}
            >
              <RefreshCw size={16} />
              Refresh
            </button>
            {r && (
              <a className="button" href="#analyst-decision">
                Record decision <ArrowRight size={16} />
              </a>
            )}
          </div>
        }
      />
      <ErrorNote text={error || report.error || ai.error} />
      {report.loading && <Loading text="Loading report…" />}
      {notice && (
        <div className="notice" role="status">
          {notice}
        </div>
      )}
      {r && (
        <div className="detail-grid">
          <div>
            <section className="panel">
              <div className="section-title">
                <h2>Report narrative</h2>
                <Tag>Source report</Tag>
              </div>
              <p className="narrative">{r.event_comments}</p>
              <div className="metadata">
                {[
                  ["Event date", r.event_date],
                  ["Location", r.location_name],
                  ["Service", r.encounter_service],
                  ["Facility", r.facility],
                  ["Age at encounter", r.age_at_encounter?.toString() || ""],
                  [
                    "Prescribed medication / dose",
                    [r.prescribed_medication, r.prescribed_dose]
                      .filter(Boolean)
                      .join(" · "),
                  ],
                  [
                    "Administered medication / dose",
                    [r.administered_medication, r.administered_dose]
                      .filter(Boolean)
                      .join(" · "),
                  ],
                  [
                    "Suspect medication / dose",
                    [r.suspect_medication, r.suspect_dose]
                      .filter(Boolean)
                      .join(" · "),
                  ],
                ].map(([k, v]) => (
                  <div key={k}>
                    <small>{k}</small>
                    <b className={!v ? "muted" : ""}>{v || "Not supplied"}</b>
                  </div>
                ))}
              </div>
            </section>
            <section className="panel">
              <div className="section-title">
                <h2>Similar historical reports</h2>
                <form
                  className="row"
                  onSubmit={(event) => {
                    event.preventDefault();
                    setN(Number(requestedN));
                  }}
                >
                  <label className="inline">
                    Results{" "}
                    <input
                      aria-label="Number of similar reports"
                      type="number"
                      min={1}
                      max={10000}
                      required
                      value={requestedN}
                      onChange={(e) => setRequestedN(e.target.value)}
                    />
                  </label>
                  <button className="secondary compact">Apply</button>
                </form>
              </div>
              <p className="muted">
                Cosine similarity describes text resemblance, not a shared
                cause.
              </p>
              <ErrorNote text={matches.error} />
              {matches.loading && (
                <Loading text="Finding historical matches…" />
              )}
              {matches.data && <Matches matches={matches.data} />}
            </section>
            <section className="panel">
              <h2>Recurring themes & trend context</h2>
              {a?.themes.map((m) => (
                <Link
                  className="theme-row"
                  to={"/trends/" + m.theme_id}
                  key={m.theme_id}
                >
                  <Layers3 size={16} />
                  {m.label}
                  <span>{m.membership_score.toFixed(3)}</span>
                  <Tag
                    kind={
                      a.trend_context.find((t) => t.theme_id === m.theme_id)
                        ?.status
                    }
                  >
                    {a.trend_context.find((t) => t.theme_id === m.theme_id)
                      ?.status || "Not evaluated"}
                  </Tag>
                </Link>
              ))}
              {!ai.loading && !ai.error && !a?.themes.length && (
                <p className="muted">
                  No theme membership above the configured threshold.
                </p>
              )}
            </section>
          </div>
          <aside>
            <section className="panel ai-panel">
              <div className="section-title">
                <h2>
                  <Activity size={18} /> AI suggestions
                </h2>
                <Tag>{a?.status || "Pending"}</Tag>
              </div>
              <p className="muted">
                Independent models. Requires human review.
              </p>
              {ai.loading && <Loading text="Loading suggestions…" />}
              {a?.severity ? (
                <>
                  <div className="grade">
                    <strong>{a.severity.grade}</strong>
                    <div>
                      Suggested PSRS grade
                      <small>{a.severity.bucket} harm bucket</small>
                    </div>
                  </div>
                  {a.severity.fixture && (
                    <Tag kind="watch">Synthetic fixture—not a prediction</Tag>
                  )}
                  <dl>
                    <dt>Raw triage score</dt>
                    <dd>{a.severity.triage_score_raw.toFixed(3)}</dd>
                    <dt>
                      {a.severity.fixture
                        ? "Fixture harm value"
                        : "Calibrated harm probability"}
                    </dt>
                    <dd>{pct(a.severity.harm_probability_calibrated)}</dd>
                    <dt>Priority review</dt>
                    <dd>
                      {a.severity.priority_flag ? "Suggested" : "Not flagged"}
                    </dd>
                  </dl>
                  <details>
                    <summary>Inspect grade probabilities</summary>
                    <Distribution
                      title="Grade probabilities"
                      values={a.severity.grade_probabilities}
                    />
                  </details>
                </>
              ) : (
                !ai.loading && (
                  <p className="unavailable">
                    Severity model unavailable or pending.
                  </p>
                )
              )}
              <hr />
              <h3>Event-type suggestion</h3>
              {a?.event_type ? (
                <>
                  <b>{a.event_type.event_type_label}</b>
                  <p>
                    {pct(a.event_type.confidence)} model confidence
                    {a.event_type.fixture ? " · FIXTURE" : ""}
                  </p>
                </>
              ) : (
                !ai.loading && (
                  <p className="unavailable">
                    Event-type model unavailable or pending.
                  </p>
                )
              )}
              <ErrorNote
                text={
                  a
                    ? Object.entries(a.errors)
                        .map(([k, v]) => `${k}: ${v}`)
                        .join(" · ")
                    : ""
                }
              />
              <button
                className="secondary"
                onClick={analyze}
                disabled={analyzing || ai.loading}
              >
                {analyzing ? "Queuing…" : "Queue analysis"}
              </button>
              <details>
                <summary>Model versions & timestamp</summary>
                <small>{a?.created_at}</small>
                {Object.entries(a?.model_versions || {}).map(([k, v]) => (
                  <p className="version" key={k}>
                    {k}: {v}
                  </p>
                ))}
              </details>
            </section>
            <section
              className="panel decision"
              id="analyst-decision"
              tabIndex={-1}
            >
              <h2>
                <CheckCircle2 size={18} /> Analyst decision
              </h2>
              <p className="muted">
                Your decision does not overwrite the AI record.
              </p>
              {r.analyst_decisions?.map((d) => (
                <div className="saved-decision" key={d.id}>
                  <b>
                    Grade {d.grade} · {d.event_type_code}
                  </b>
                  <p>{d.note || "No note supplied"}</p>
                  <small>{new Date(d.created_at).toLocaleString()}</small>
                </div>
              ))}
              <form className="form" onSubmit={feedback} aria-busy={saving}>
                <div className="form-grid">
                  <label>
                    Confirmed grade
                    <select name="grade" required defaultValue="">
                      <option value="" disabled>
                        Select grade
                      </option>
                      {grades.map((g) => (
                        <option key={g}>{g}</option>
                      ))}
                    </select>
                  </label>
                  <label>
                    Confirmed type
                    <select name="event_type_code" required defaultValue="">
                      <option value="" disabled>
                        Select event type
                      </option>
                      {codes.map((c) => (
                        <option key={c}>{c}</option>
                      ))}
                    </select>
                  </label>
                </div>
                <label>
                  Analyst note
                  <textarea name="note" maxLength={2000} />
                </label>
                <button disabled={saving}>
                  {saving ? "Saving decision…" : "Record analyst decision"}
                </button>
              </form>
            </section>
          </aside>
        </div>
      )}
    </>
  );
}
function Distribution({
  title,
  values,
}: {
  title: string;
  values: Record<string, number>;
}) {
  return (
    <div className="distribution">
      <h3>{title}</h3>
      {Object.entries(values).map(([label, value]) => (
        <div key={label}>
          <span>{label}</span>
          <div className="track">
            <i
              style={{
                width: `${(value / Math.max(...Object.values(values), 0.001)) * 100}%`,
              }}
            />
          </div>
          <b>{Number.isInteger(value) ? value : value.toFixed(2)}</b>
        </div>
      ))}
    </div>
  );
}
function Trends() {
  const [tick, setTick] = useState(0),
    [notice, setNotice] = useState(""),
    [error, setError] = useState(""),
    [busy, setBusy] = useState(false);
  const patterns = useLoad<Pattern[]>("/patterns", tick);
  const health = useLoad<Health>("/health", tick);
  async function rebuild() {
    if (busy) return;
    setBusy(true);
    setError("");
    setNotice("");
    try {
      await api("/patterns/rebuild", {});
      setNotice("Rebuild queued. Refresh to check completed patterns.");
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  return (
    <>
      <Heading
        eyebrow="PATTERN DISCOVERY"
        title="Patterns & trends"
        description="Multi-label semantic themes with explainable, weekly trend comparisons."
        action={
          <div className="row">
            <button
              className="secondary"
              onClick={() => {
                setTick(tick + 1);
                setNotice("");
                setError("");
              }}
            >
              <RefreshCw size={16} />
              Refresh
            </button>
          </div>
        }
      />
      <ErrorNote text={error || patterns.error || health.error} />
      {health.data && health.data.rebuild.status !== "idle" && (
        <p className="muted">Pattern rebuild: {health.data.rebuild.status}</p>
      )}
      <ErrorNote text={health.data?.rebuild.error || ""} />
      {notice && (
        <div className="notice" role="status">
          {notice}
        </div>
      )}
      {patterns.loading && <Loading text="Loading patterns…" />}
      <div className="pattern-grid">
        {patterns.data
          ?.slice()
          .sort(patternOrder)
          .map((p) => (
            <PatternCard key={p.theme_id} pattern={p} />
          ))}
      </div>
      {patterns.data?.length === 0 && (
        <Empty text="No recurring patterns. Add and analyze reports, then rebuild." />
      )}
      <details className="panel maintenance">
        <summary>Pattern maintenance</summary>
        <p>
          Rebuild after adding reports to discover new themes. Existing theme
          membership and theme links may change.
        </p>
        <button
          className="secondary"
          disabled={
            busy ||
            health.loading ||
            ["queued", "running"].includes(health.data?.rebuild.status || "")
          }
          onClick={rebuild}
        >
          <Layers3 size={16} />
          {busy ? "Queuing rebuild…" : "Rebuild patterns"}
        </button>
      </details>
      <div className="footnote">
        Counts compare the last complete week with prior weeks, including
        zero-count weeks. Trends indicate changes in reporting, not causality.
      </div>
    </>
  );
}
function TrendDetail() {
  const { id } = useParams();
  const pattern = useLoad<Pattern>("/patterns/" + id),
    events = useLoad<{ items: Event[]; total: number }>(
      "/patterns/" + id + "/events",
    );
  const p = pattern.data;
  return (
    <>
      <Link className="back" to="/trends">
        ← All patterns
      </Link>
      <Heading
        eyebrow="RECURRING THEME"
        title={
          p?.name ||
          (pattern.error ? "Pattern unavailable" : "Loading pattern…")
        }
        description="Explore the reports behind the signal. Membership can overlap other themes."
      />
      <ErrorNote text={pattern.error || events.error} />
      {pattern.loading && <Loading text="Loading pattern…" />}
      {p && (
        <>
          <div className="detail-grid">
            <section className={"panel " + (p.trend?.status || "")}>
              <div className="section-title">
                <h2>Weekly report timeline</h2>
                <Tag kind={p.trend?.status}>{p.trend?.status}</Tag>
              </div>
              {p.trend && (
                <>
                  <Spark trend={p.trend} large />
                  <p>
                    {p.trend.current_count} events in {p.trend.period_start}–
                    {p.trend.period_end} · baseline{" "}
                    {p.trend.baseline_count.toFixed(1)} per week ·{" "}
                    {change(p.trend)}
                  </p>
                </>
              )}
              <div className="row">
                {p.keywords.map((k) => (
                  <Tag key={k}>{k}</Tag>
                ))}
              </div>
            </section>
            <section className="panel">
              <h2>Theme overview</h2>
              <dl>
                <dt>Associated reports</dt>
                <dd>{p.event_count}</dd>
                <dt>First seen</dt>
                <dd>{p.first_seen}</dd>
                <dt>Last seen</dt>
                <dd>{p.last_seen}</dd>
              </dl>
              <h3>Representative cases</h3>
              {p.representative_event_ids.map((key, i) => (
                <Link className="theme-row" key={key} to={"/events/" + key}>
                  Representative report {i + 1}
                  <ArrowUpRight size={16} />
                </Link>
              ))}
            </section>
          </div>
          <div className="three-grid">
            <section className="panel">
              <Distribution
                title="AI event types"
                values={p.event_type_distribution}
              />
            </section>
            <section className="panel">
              <Distribution
                title="AI severity grades"
                values={p.severity_distribution}
              />
            </section>
            <section className="panel">
              <Distribution
                title="Locations"
                values={p.location_distribution}
              />
            </section>
          </div>
          <div className="section-title">
            <h2>Associated cases</h2>
            <span>{events.data?.total || 0} total · first 100 shown</span>
          </div>
          <EventTable
            events={events.data?.items || []}
            loading={events.loading}
            error={events.error}
          />
        </>
      )}
    </>
  );
}
function SearchPage() {
  const [query, setQuery] = useState(""),
    [n, setN] = useState(5),
    [matches, setMatches] = useState<Match[] | null>(null),
    [busy, setBusy] = useState(false),
    [error, setError] = useState("");
  async function search(e: React.FormEvent) {
    e.preventDefault();
    if (busy) return;
    setBusy(true);
    setError("");
    setMatches(null);
    try {
      setMatches(await api("/search", { query, top_n: n }));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  return (
    <>
      <Heading
        eyebrow="CASE LINKAGE"
        title="Find similar reports"
        description="Search local reports by meaning. Choose how many matches to retrieve."
      />
      <section className="panel search-panel">
        <form onSubmit={search} className="form" aria-busy={busy}>
          <fieldset disabled={busy}>
            <label>
              Describe the event
              <textarea
                required
                minLength={3}
                value={query}
                onChange={(e) => {
                  setQuery(e.target.value);
                  setMatches(null);
                }}
                placeholder="A medication was nearly administered at the wrong dose after a barcode scanner problem."
              />
            </label>
            <div className="row">
              <label className="inline">
                Top N matches
                <input
                  type="number"
                  min={1}
                  max={10000}
                  required
                  value={n}
                  onChange={(e) => {
                    setN(Number(e.target.value));
                    setMatches(null);
                  }}
                />
              </label>
              <button disabled={busy}>
                <Search size={16} />
                {busy ? "Searching…" : "Find similar reports"}
              </button>
            </div>
          </fieldset>
        </form>
        <p className="muted">
          <LockKeyhole size={13} /> Query text stays in the local request body,
          never the URL.
        </p>
        <ErrorNote text={error} />
        {busy && <Loading text="Searching local reports…" />}
        {matches && (
          <section aria-label="Search results">
            <div role="status">
              <h2>{matches.length} matching reports</h2>
            </div>
            <p className="muted">
              Sorted by cosine similarity, not likelihood of a shared cause.
            </p>
            <Matches matches={matches} historical={false} />
          </section>
        )}
      </section>
    </>
  );
}
export function App() {
  const { pathname } = useLocation();
  const [tick, setTick] = useState(0);
  const health = useLoad<Health>("/health", tick);
  return (
    <div className="app">
      <a className="skip-link" href="#main-content">
        Skip to main content
      </a>
      <RouteFocus />
      <aside className="sidebar">
        <Link className="brand" to="/dashboard">
          <span>
            <Activity size={26} />
          </span>
          <div>
            MIDAS<small>SAFETY INTELLIGENCE</small>
          </div>
        </Link>
        <div className="workspace-label">ANALYST WORKSPACE</div>
        <nav aria-label="Main navigation">
          {[
            [LayoutDashboard, "Overview", "/dashboard"],
            [ClipboardList, "Event worklist", "/worklist"],
            [BarChart3, "Patterns & trends", "/trends"],
            [Search, "Similar-case search", "/search"],
          ].map(([Icon, label, to]) => {
            const I = Icon as typeof Search;
            return (
              <NavLink
                key={to as string}
                to={to as string}
                className={({ isActive }) =>
                  isActive ||
                  (to === "/worklist" && pathname.startsWith("/events/")) ||
                  (to === "/dashboard" && pathname === "/")
                    ? "active"
                    : ""
                }
              >
                <I size={18} />
                {label as string}
              </NavLink>
            );
          })}
        </nav>
        <div className="sidebar-bottom">
          <div>
            <span className="status-dot" />
            Local workspace
          </div>
          <p>Reports stay on this device.</p>
          <div className="project-label">
            PENN STATE HEALTH
            <br />
            <small>Capstone prototype · Unofficial</small>
          </div>
        </div>
      </aside>
      <div className="main">
        <header>
          <span>
            Hospital safety <ChevronRight size={14} /> Analyst workspace
          </span>
          <span>
            <LockKeyhole size={14} /> Local only{" "}
            <span className="avatar">SA</span>
          </span>
        </header>
        <div
          className={
            "mode-banner " +
            (health.error
              ? "connection-error"
              : health.data?.fixture_mode
                ? "fixture"
                : "")
          }
        >
          <ShieldCheck size={16} />
          {health.error
            ? "Backend unavailable. Start the local API."
            : health.loading
              ? "Checking local model status…"
              : health.data?.fixture_mode
                ? "DEVELOPMENT FIXTURES · Severity and event type are synthetic examples, not medical predictions."
                : Object.keys(health.data?.errors || {}).length
                  ? "Some local models are unavailable. Reports can still be saved and reviewed."
                  : "Local model mode · AI suggestions require analyst review."}
          <button
            className="icon-button"
            onClick={() => setTick(tick + 1)}
            aria-label="Refresh model status"
          >
            <RefreshCw size={14} />
          </button>
        </div>
        {health.data?.models.pattern.startsWith("LEXICAL") && (
          <div className="mode-banner">
            Lexical fallback active · Not MiniLM semantic embeddings
          </div>
        )}
        <main id="main-content" tabIndex={-1}>
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/dashboard" element={<Dashboard />} />
            <Route path="/worklist" element={<Worklist />} />
            <Route path="/events/:id" element={<EventDetail />} />
            <Route path="/trends" element={<Trends />} />
            <Route path="/trends/:id" element={<TrendDetail />} />
            <Route path="/search" element={<SearchPage />} />
            <Route path="*" element={<Empty text="Page not found" />} />
          </Routes>
        </main>
      </div>
    </div>
  );
}
