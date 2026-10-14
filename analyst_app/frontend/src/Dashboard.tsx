import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import {
  AlertTriangle,
  ArrowRight,
  CheckCircle2,
  ClipboardList,
  Layers3,
  LoaderCircle,
  Plus,
  RefreshCw,
  Search,
} from "lucide-react";
import type { DashboardData, DashboardReport, Pattern } from "./types";

function useOverview() {
  const [data, setData] = useState<DashboardData | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(true);
  const [revision, setRevision] = useState(0);
  useEffect(() => {
    const controller = new AbortController();
    let live = true;
    let timedOut = false;
    const timeout = window.setTimeout(() => {
      timedOut = true;
      controller.abort();
    }, 15000);
    setBusy(true);
    setError("");
    async function load() {
      try {
        const result = await fetch("/api/dashboard", {
          signal: controller.signal,
        });
        if (!result.ok) throw new Error("Dashboard request failed.");
        const value: DashboardData = await result.json();
        // Fail clearly against an old server instead of rendering misleading zeros.
        if (
          ![
            value.event_count,
            value.unreviewed_count,
            value.reviewed_count,
            value.priority_unreviewed_count,
            value.analysis_issue_count,
            value.pending_analysis_count,
          ].every((n) => Number.isInteger(n) && n >= 0) ||
          !Array.isArray(value.review_queue) ||
          !Array.isArray(value.recent_events) ||
          !Array.isArray(value.patterns) ||
          !Number.isFinite(Date.parse(value.generated_at))
        ) {
          throw new Error(
            "Dashboard response is incompatible. Restart the updated local API.",
          );
        }
        if (live) setData(value);
      } catch (e) {
        if (live)
          setError(
            timedOut
              ? "The local API took too long to respond. Try again."
              : e instanceof Error && e.message.startsWith("Dashboard response")
                ? e.message
                : "Could not refresh the dashboard. Check the local API, then try again.",
          );
      } finally {
        window.clearTimeout(timeout);
        if (live) setBusy(false);
      }
    }
    void load();
    return () => {
      live = false;
      window.clearTimeout(timeout);
      controller.abort();
    };
  }, [revision]);
  return { data, error, busy, refresh: () => setRevision((r) => r + 1) };
}

function PatternSummary({ pattern }: { pattern: Pattern }) {
  const trend = pattern.trend;
  const status = pattern.trend_stale
    ? "Past period"
    : trend?.status || "Not evaluated";
  return (
    <Link className="dashboard-pattern" to={"/trends/" + pattern.theme_id}>
      <div className="section-title">
        <span
          className={"tag " + (!pattern.trend_stale ? trend?.status || "" : "")}
        >
          {status}
        </span>
        <ArrowRight size={15} />
      </div>
      <h3>{pattern.name}</h3>
      <div className="muted">{pattern.event_count} associated reports</div>
      {trend && (
        <p>
          <strong>{trend.current_count}</strong> in the week ending{" "}
          {trend.period_end}
          <br />
          Baseline {trend.baseline_count.toFixed(1)} / week
        </p>
      )}
    </Link>
  );
}

function Queue({ events }: { events: DashboardReport[] }) {
  return (
    <div
      className="table-wrap dashboard-table"
      role="region"
      aria-label="Dashboard reports"
      tabIndex={0}
    >
      <table>
        <caption className="sr-only">
          Reports for analyst review. Priority signals are AI suggestions, not
          final decisions.
        </caption>
        <thead>
          <tr>
            <th scope="col">Report / location</th>
            <th scope="col">Event date</th>
            <th scope="col">Review status</th>
            <th scope="col">AI signal</th>
          </tr>
        </thead>
        <tbody>
          {events.map((event) => {
            const severity = event.ai?.severity;
            return (
              <tr key={event.id}>
                <td>
                  <Link className="event-link" to={"/events/" + event.id}>
                    {event.event_id}
                    <ArrowRight size={14} />
                  </Link>
                  <small>
                    {event.location_name || "Location not supplied"}
                  </small>
                </td>
                <td>{event.event_date}</td>
                <td>
                  <span className="review-label">
                    {event.review_status === "reviewed" ? (
                      <>
                        <CheckCircle2 size={14} />
                        Reviewed
                      </>
                    ) : (
                      "Awaiting review"
                    )}
                  </span>
                </td>
                <td>
                  {severity?.priority_flag ? (
                    <span className="tag watch">
                      <AlertTriangle size={13} />
                      {severity.fixture
                        ? "Fixture priority"
                        : "Priority suggested"}
                    </span>
                  ) : (
                    <span className="muted">
                      {!event.ai || event.ai.status === "pending"
                        ? "Pending"
                        : event.ai.status === "failed"
                          ? "Unavailable"
                          : severity
                            ? "No priority flag"
                            : "Severity unavailable"}
                    </span>
                  )}
                  {event.ai?.status === "partial" && (
                    <small>Some analysis unavailable</small>
                  )}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

export function Dashboard() {
  const { data, error, busy, refresh } = useOverview();
  const [view, setView] = useState<"queue" | "recent">("queue");
  const count = (value?: number) =>
    value === undefined ? "—" : value.toLocaleString();
  const cards = [
    {
      label: "Awaiting review",
      value: data?.unreviewed_count,
      note: "No analyst decision recorded",
      filter: "unreviewed",
      Icon: ClipboardList,
    },
    {
      label: "Priority · awaiting review",
      value: data?.priority_unreviewed_count,
      note: data?.fixture_mode
        ? "Synthetic classifier suggestions"
        : "AI suggestions, not final decisions",
      filter: "priority",
      Icon: AlertTriangle,
    },
    {
      label: "Analysis issues",
      value: data?.analysis_issue_count,
      note: "Partial or failed AI analysis",
      filter: "analysis_issues",
      Icon: Layers3,
    },
    {
      label: "Reviewed",
      value: data?.reviewed_count,
      note: "At least one analyst decision",
      filter: "reviewed",
      Icon: CheckCircle2,
    },
  ];
  const reports = data
    ? view === "queue"
      ? data.review_queue
      : data.recent_events
    : [];
  const patterns =
    data?.patterns
      .slice()
      .sort((a, b) => {
        const rank = (p: Pattern) =>
          p.trend_stale
            ? 4
            : p.trend?.status === "emerging"
              ? 0
              : p.trend?.status === "watch"
                ? 1
                : 2;
        return rank(a) - rank(b) || b.event_count - a.event_count;
      })
      .slice(0, 3) || [];
  const stalePatterns = data?.patterns.some((p) => p.trend_stale);
  return (
    <div className="dashboard">
      <div className="heading">
        <div>
          <div className="eyebrow">ANALYST WORKSPACE</div>
          <h1>Safety overview</h1>
          <p>
            Start with reports awaiting review. Explore patterns for context.
          </p>
        </div>
        <div className="row">
          <button className="secondary" disabled={busy} onClick={refresh}>
            <RefreshCw size={16} className={busy ? "spin" : ""} />
            {busy ? "Refreshing…" : "Refresh"}
          </button>
          <Link className="button" to="/worklist?new=1">
            <Plus size={16} />
            New report
          </Link>
        </div>
      </div>
      <div className="dashboard-meta" aria-live="polite">
        <span>
          {data
            ? `${count(data.event_count)} reports in this workspace`
            : "Loading workspace…"}
        </span>
        {data && (
          <span>
            {error ? "Last successful update" : "Updated"}{" "}
            <time dateTime={data.generated_at}>
              {new Date(data.generated_at).toLocaleString()}
            </time>{" "}
            · refresh manually
          </span>
        )}
      </div>
      {error && (
        <div className="error dashboard-error" role="alert">
          <div>
            <strong>
              {data
                ? "Showing previously loaded data."
                : "Dashboard unavailable."}
            </strong>
            <br />
            {error}
          </div>
          <button className="secondary" onClick={refresh} disabled={busy}>
            Try again
          </button>
        </div>
      )}
      <div
        className="metrics dashboard-metrics"
        aria-label="Workspace totals"
        aria-busy={busy}
      >
        {cards.map(({ label, value, note, filter, Icon }) => (
          <Link
            className={
              "metric " +
              ((filter === "priority" || filter === "analysis_issues") &&
              Number(value) > 0
                ? "attention"
                : "")
            }
            to={"/worklist?filter=" + filter}
            key={filter}
          >
            <span>
              <Icon size={17} />
              {label}
            </span>
            <strong>{count(value)}</strong>
            <small>{note}</small>
            <span className="metric-action">
              View reports <ArrowRight size={13} />
            </span>
          </Link>
        ))}
      </div>
      {data && data.pending_analysis_count > 0 && (
        <div className="dashboard-pending">
          <LoaderCircle size={15} />
          <span>
            {data.pending_analysis_count} reports have pending analysis. They
            remain available for human review. Refresh to check progress.
          </span>
        </div>
      )}
      <div className="dashboard-columns">
        <section className="panel queue-panel" aria-labelledby="queue-heading">
          <div className="section-title">
            <div>
              <h2 id="queue-heading">
                {view === "queue" ? "Review queue" : "Recent reports"}
              </h2>
              <p>
                {view === "queue"
                  ? "Priority suggestions first, then newest event date."
                  : "Newest event dates, including reviewed reports."}
              </p>
            </div>
          </div>
          <div className="queue-controls">
            <div className="segmented" role="group" aria-label="Report view">
              <button
                className="secondary"
                aria-pressed={view === "queue"}
                onClick={() => setView("queue")}
              >
                Awaiting review
              </button>
              <button
                className="secondary"
                aria-pressed={view === "recent"}
                onClick={() => setView("recent")}
              >
                Recent reports
              </button>
            </div>
            <Link
              className="text-link"
              to={
                view === "queue" ? "/worklist?filter=unreviewed" : "/worklist"
              }
            >
              Open worklist <ArrowRight size={14} />
            </Link>
          </div>
          {!data ? (
            <div className="empty">
              {busy ? (
                <div role="status">Loading review queue…</div>
              ) : (
                <p>Reports could not be loaded. Retry above.</p>
              )}
            </div>
          ) : reports.length ? (
            <>
              <Queue events={reports} />
              <div className="queue-caption">
                Showing {reports.length} of{" "}
                {view === "queue" ? data.unreviewed_count : data.event_count}{" "}
                reports. Open the worklist for the complete list.
              </div>
            </>
          ) : (
            <div className="empty">
              <CheckCircle2 size={26} />
              <h3>
                {data.event_count === 0
                  ? "Your workspace is ready"
                  : "Review queue is clear"}
              </h3>
              <p>
                {data.event_count === 0
                  ? "Add an authored synthetic report to begin."
                  : "Every report has at least one analyst decision. You can still revisit prior reviews."}
              </p>
              <Link
                className="button secondary"
                to={
                  data.event_count === 0
                    ? "/worklist?new=1"
                    : "/worklist?filter=reviewed"
                }
              >
                {data.event_count === 0
                  ? "Add first report"
                  : "View reviewed reports"}
              </Link>
            </div>
          )}
        </section>
        <aside className="dashboard-context" aria-label="Pattern context">
          <section className="panel">
            <div className="section-title">
              <h2>Pattern context</h2>
              <Link to="/trends">
                View all <ArrowRight size={14} />
              </Link>
            </div>
            <p className="muted">Exploratory themes, not confirmed causes.</p>
            {stalePatterns && (
              <div className="stale-note">
                Some trend periods are out of date. Open Patterns & trends to
                rebuild before interpreting them as current signals.
              </div>
            )}
            {!data ? (
              <p className="muted">
                {busy ? "Loading patterns…" : "Patterns unavailable."}
              </p>
            ) : patterns.length ? (
              patterns.map((p) => (
                <PatternSummary key={p.theme_id} pattern={p} />
              ))
            ) : (
              <div className="pattern-empty">
                <p>No themes yet. Analyze reports, then rebuild patterns.</p>
                <Link className="text-link" to="/trends">
                  Open pattern tools <ArrowRight size={14} />
                </Link>
              </div>
            )}
          </section>
          <Link className="search-shortcut" to="/search">
            <Search size={20} />
            <span>
              <strong>Find a similar case</strong>
              <small>Search local reports by meaning</small>
            </span>
            <ArrowRight size={16} />
          </Link>
        </aside>
      </div>
      <p className="footnote">
        AI signals support review; they do not determine the analyst’s decision.
        “Reviewed” means a decision has been recorded, not that an event is
        resolved.
      </p>
    </div>
  );
}
