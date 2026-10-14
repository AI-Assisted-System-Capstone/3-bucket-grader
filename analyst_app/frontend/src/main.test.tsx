import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { App } from "./main";

// Authored fixtures only. No server, model, hospital input, or network required.
const record = {
  id: "test-event",
  event_id: "SYNTHETIC-UX-01",
  event_comments: "Authored synthetic scanner test report.",
  event_date: "2026-10-13",
  location_name: "Synthetic unit",
  facility: "Test campus",
  encounter_service: "",
  age_at_encounter: null,
  prescribed_medication: "",
  prescribed_dose: "",
  administered_medication: "",
  administered_dose: "",
  suspect_medication: "",
  suspect_dose: "",
};
const ai = {
  event_id: record.id,
  status: "complete",
  themes: [],
  trend_context: [],
  similar_events: [],
  errors: {},
  model_versions: {},
  created_at: "2026-10-13T12:00:00Z",
  severity: {
    grade: "E",
    bucket: "some",
    triage_score_raw: 0.65,
    harm_probability_calibrated: 0.2,
    priority_flag: true,
    fixture: true,
    grade_probabilities: { E: 0.8, A: 0.2 },
  },
  event_type: {
    event_type_label: "Medication error",
    confidence: 0.8,
    fixture: true,
  },
};
const pattern = (id: string, status: string) => ({
  theme_id: id,
  name: id + " theme",
  keywords: [],
  event_count: 3,
  location_distribution: {},
  representative_event_ids: [],
  trend: {
    status,
    current_count: 3,
    baseline_count: 1,
    percent_change: 200,
    timeline: [],
    period_end: "2026-10-11",
  },
});
let decisions: unknown[];
let fetchMock: ReturnType<typeof vi.fn>;
function response(value: unknown, status = 200) {
  return { ok: status < 400, status, json: async () => value } as Response;
}
function resolve(path: string, init?: RequestInit) {
  if (path === "/api/health")
    return {
      fixture_mode: true,
      models: { pattern: "all-MiniLM" },
      errors: {},
      rebuild: { status: "idle" },
    };
  if (path === "/api/dashboard")
    return {
      event_count: 1,
      high_priority_count: 1,
      emerging_pattern_count: 1,
      generated_at: "2026-10-13T12:00:00Z",
      fixture_mode: true,
      unreviewed_count: 1,
      reviewed_count: 0,
      priority_unreviewed_count: 1,
      analysis_issue_count: 0,
      pending_analysis_count: 0,
      review_queue: [{ ...record, ai, review_status: "unreviewed" }],
      recent_events: [{ ...record, ai, review_status: "unreviewed" }],
      patterns: [pattern("stable", "stable"), pattern("emerging", "emerging")],
    };
  if (path === "/api/patterns")
    return [pattern("stable", "stable"), pattern("emerging", "emerging")];
  if (path.startsWith("/api/events?"))
    return { total: 1, items: [{ ...record, ai }] };
  if (path === "/api/events" && init?.method === "POST") return record;
  if (path.endsWith("/feedback")) {
    decisions.push({
      id: "decision-1",
      ...JSON.parse(init!.body as string),
      created_at: "2026-10-13T12:00:00Z",
    });
    return { status: "saved" };
  }
  if (path === "/api/events/test-event")
    return { ...record, analyst_decisions: decisions };
  if (path.endsWith("/ai")) return ai;
  if (path.includes("/similar?")) return [];
  if (path === "/api/search")
    return [
      {
        event_id: record.id,
        source_event_id: record.event_id,
        similarity: 0.9,
        event_date: record.event_date,
        location_name: record.location_name,
      },
    ];
  if (path === "/api/patterns/rebuild") return { status: "queued" };
  throw new Error("Unexpected test route: " + path);
}
function open(route: string) {
  render(
    <MemoryRouter initialEntries={[route]}>
      <App />
    </MemoryRouter>,
  );
}
beforeEach(() => {
  decisions = [];
  vi.stubGlobal("scrollTo", vi.fn());
  fetchMock = vi.fn(async (path: string, init?: RequestInit) =>
    response(resolve(path, init)),
  );
  vi.stubGlobal("fetch", fetchMock);
});
afterEach(() => {
  cleanup();
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe("analyst UX regression checks", () => {
  it("shows fixture disclosures, prioritizes emerging themes, and exposes keyboard navigation", async () => {
    open("/dashboard");
    await screen.findByText(/DEVELOPMENT FIXTURES/);
    await screen.findByRole("link", { name: record.event_id });
    const cards = document.querySelectorAll(".dashboard-pattern");
    expect(cards[0].textContent).toContain("emerging theme");
    expect(cards[1].textContent).toContain("stable theme");
    expect(document.querySelectorAll(".metric.attention")).toHaveLength(1);
    expect(
      screen
        .getByRole("link", { name: "Skip to main content" })
        .getAttribute("href"),
    ).toBe("#main-content");
    expect(
      screen.getByRole("navigation", { name: "Main navigation" }),
    ).toBeTruthy();
    expect(screen.getByText("Fixture priority")).toBeTruthy();
  });

  it("does not misrepresent an in-flight load as an empty workspace", async () => {
    fetchMock.mockImplementation(() => new Promise(() => {}));
    open("/worklist");
    expect(screen.getByText("Loading reports…")).toBeTruthy();
    expect(screen.queryByText(/No reports yet/)).toBeNull();
    expect(screen.getByText("Checking local model status…")).toBeTruthy();
  });

  it("keeps the last dashboard snapshot visible when refresh fails and recovers", async () => {
    open("/dashboard");
    await screen.findByRole("link", { name: record.event_id });
    fetchMock.mockImplementation(async (path: string, init?: RequestInit) =>
      path === "/api/dashboard"
        ? response({}, 503)
        : response(resolve(path, init)),
    );
    await userEvent.click(screen.getByRole("button", { name: "Refresh" }));
    await screen.findByText("Showing previously loaded data.");
    expect(screen.getByRole("link", { name: record.event_id })).toBeTruthy();
    expect(screen.getByText(/Last successful update/)).toBeTruthy();
    fetchMock.mockImplementation(async (path: string, init?: RequestInit) =>
      response(resolve(path, init)),
    );
    await userEvent.click(screen.getByRole("button", { name: "Try again" }));
    await waitFor(() => expect(screen.queryByRole("alert")).toBeNull());
  });

  it("does not mistake a failed initial dashboard request for an empty workspace", async () => {
    fetchMock.mockImplementation(async (path: string, init?: RequestInit) =>
      path === "/api/dashboard"
        ? response({}, 500)
        : response(resolve(path, init)),
    );
    open("/dashboard");
    await screen.findByText("Dashboard unavailable.");
    expect(screen.queryByText("Your workspace is ready")).toBeNull();
    expect(screen.queryByText("Review queue is clear")).toBeNull();
    expect(document.querySelector(".metric strong")?.textContent).toBe("—");
  });

  it("offers a real first-report action when the dashboard is empty", async () => {
    fetchMock.mockImplementation(async (path: string, init?: RequestInit) =>
      response(
        path === "/api/dashboard"
          ? {
              ...(resolve(path) as object),
              event_count: 0,
              unreviewed_count: 0,
              priority_unreviewed_count: 0,
              review_queue: [],
              recent_events: [],
              patterns: [],
            }
          : resolve(path, init),
      ),
    );
    open("/dashboard");
    await screen.findByText("Your workspace is ready");
    await userEvent.click(
      screen.getByRole("link", { name: "Add first report" }),
    );
    expect(
      await screen.findByRole("heading", { name: "Submit a synthetic report" }),
    ).toBeTruthy();
  });

  it("separates a completed review queue from recent reports and stale trends", async () => {
    fetchMock.mockImplementation(async (path: string, init?: RequestInit) =>
      response(
        path === "/api/dashboard"
          ? {
              ...(resolve(path) as object),
              unreviewed_count: 0,
              reviewed_count: 1,
              priority_unreviewed_count: 0,
              review_queue: [],
              recent_events: [{ ...record, ai, review_status: "reviewed" }],
              patterns: [{ ...pattern("old", "emerging"), trend_stale: true }],
            }
          : resolve(path, init),
      ),
    );
    open("/dashboard");
    await screen.findByText("Review queue is clear");
    expect(screen.getByText("Past period")).toBeTruthy();
    expect(screen.getByText(/Some trend periods are out of date/)).toBeTruthy();
    await userEvent.click(
      screen.getByRole("button", { name: "Recent reports" }),
    );
    expect(
      await screen.findByRole("link", { name: record.event_id }),
    ).toBeTruthy();
    expect(screen.queryByText("Review queue is clear")).toBeNull();
  });

  it("opens the matching global worklist filter from a dashboard metric", async () => {
    open("/dashboard");
    await screen.findByRole("link", { name: record.event_id });
    await userEvent.click(
      screen.getByRole("link", { name: /Priority · awaiting review/ }),
    );
    expect(
      (screen.getByLabelText("Show reports") as HTMLSelectElement).value,
    ).toBe("priority");
    await waitFor(() =>
      expect(
        fetchMock.mock.calls.some(
          ([path]) => path === "/api/events?offset=0&limit=50&filter=priority",
        ),
      ).toBe(true),
    );
  });

  it("resets pagination when a worklist filter changes", async () => {
    open("/worklist?filter=reviewed&page=2");
    await screen.findByRole("link", { name: record.event_id });
    await userEvent.selectOptions(
      screen.getByLabelText("Show reports"),
      "unreviewed",
    );
    await waitFor(() =>
      expect(
        fetchMock.mock.calls.some(
          ([path]) =>
            path === "/api/events?offset=0&limit=50&filter=unreviewed",
        ),
      ).toBe(true),
    );
  });

  it("handles an incompatible dashboard response without crashing", async () => {
    fetchMock.mockImplementation(async (path: string, init?: RequestInit) =>
      response(
        path === "/api/dashboard" ? { event_count: 4 } : resolve(path, init),
      ),
    );
    open("/dashboard");
    await screen.findByText(/Dashboard response is incompatible/);
    expect(screen.getByRole("button", { name: "Try again" })).toBeTruthy();
  });

  it("times out a hung dashboard request and makes retry available", async () => {
    vi.useFakeTimers();
    fetchMock.mockImplementation((path: string, init?: RequestInit) =>
      path === "/api/dashboard"
        ? new Promise((_resolve, reject) => {
            init?.signal?.addEventListener("abort", () =>
              reject(new DOMException("Aborted", "AbortError")),
            );
          })
        : Promise.resolve(response(resolve(path, init))),
    );
    open("/dashboard");
    await act(async () => {
      await vi.advanceTimersByTimeAsync(15000);
    });
    expect(
      screen.getByText("The local API took too long to respond. Try again."),
    ).toBeTruthy();
    expect(
      (screen.getByRole("button", { name: "Try again" }) as HTMLButtonElement)
        .disabled,
    ).toBe(false);
  });

  it("shows a useful API error and recovers through Refresh", async () => {
    let fail = true;
    fetchMock.mockImplementation(async (path: string, init?: RequestInit) =>
      path.startsWith("/api/events?") && fail
        ? response({}, 500)
        : response(resolve(path, init)),
    );
    open("/worklist");
    expect(await screen.findByRole("alert")).toBeTruthy();
    expect(screen.queryByText(/No reports yet/)).toBeNull();
    fail = false;
    await userEvent.click(screen.getByRole("button", { name: "Refresh" }));
    await screen.findByRole("link", { name: record.event_id });
    expect(screen.queryByRole("alert")).toBeNull();
  });

  it("uses a POST body with configurable top N and clears stale search results", async () => {
    open("/search");
    const user = userEvent.setup();
    await user.type(
      screen.getByLabelText("Describe the event"),
      "Synthetic scanner issue",
    );
    await user.clear(screen.getByLabelText("Top N matches"));
    await user.type(screen.getByLabelText("Top N matches"), "12");
    await user.click(
      screen.getByRole("button", { name: "Find similar reports" }),
    );
    await screen.findByRole("heading", { name: "1 matching reports" });
    const call = fetchMock.mock.calls.find(([path]) => path === "/api/search")!;
    expect(call[1].method).toBe("POST");
    expect(JSON.parse(call[1].body)).toEqual({
      query: "Synthetic scanner issue",
      top_n: 12,
    });
    await user.type(screen.getByLabelText("Describe the event"), " changed");
    expect(screen.queryByRole("region", { name: "Search results" })).toBeNull();
  });

  it("blocks duplicate report saves while in flight and opens the saved report", async () => {
    let finish: (value: Response) => void = () => {};
    fetchMock.mockImplementation((path: string, init?: RequestInit) =>
      path === "/api/events"
        ? new Promise<Response>((r) => {
            finish = r;
          })
        : Promise.resolve(response(resolve(path, init))),
    );
    const user = userEvent.setup();
    open("/worklist");
    await user.click(screen.getByRole("button", { name: "New report" }));
    await user.type(
      screen.getByLabelText("Source event ID *"),
      record.event_id,
    );
    await user.type(screen.getByLabelText("Event date *"), record.event_date);
    await user.type(
      screen.getByLabelText("Report narrative *"),
      record.event_comments,
    );
    await user.click(
      screen.getByRole("button", { name: "Save report & queue analysis" }),
    );
    const pending = screen.getByRole("button", {
      name: "Saving report…",
    }) as HTMLButtonElement;
    expect(pending.disabled).toBe(true);
    await user.click(pending);
    expect(
      fetchMock.mock.calls.filter(([path]) => path === "/api/events"),
    ).toHaveLength(1);
    finish(response(record));
    await screen.findByRole("heading", { name: record.event_id });
    expect(
      screen.getByRole("link", { name: "Event worklist" }).className,
    ).toContain("active");
  });

  it("keeps human decisions distinct from suggestions and requires explicit selections", async () => {
    open("/events/test-event");
    await screen.findByRole("heading", { name: record.event_id });
    const user = userEvent.setup();
    expect(
      (screen.getByLabelText("Confirmed grade") as HTMLSelectElement).value,
    ).toBe("");
    expect(
      (screen.getByLabelText("Confirmed grade") as HTMLSelectElement).required,
    ).toBe(true);
    await user.selectOptions(screen.getByLabelText("Confirmed grade"), "B2");
    await user.selectOptions(screen.getByLabelText("Confirmed type"), "ME");
    await user.click(
      screen.getByRole("button", { name: "Record analyst decision" }),
    );
    await screen.findByText("Grade B2 · ME");
    expect(document.querySelector(".grade > strong")?.textContent).toBe("E");
    expect(screen.getByText("Synthetic fixture—not a prediction")).toBeTruthy();
    expect(
      screen.getByText("Inspect grade probabilities").closest("details")?.open,
    ).toBe(false);
  });

  it("applies similar-report count deliberately instead of on every keystroke", async () => {
    open("/events/test-event");
    await screen.findByLabelText("Number of similar reports");
    const user = userEvent.setup();
    await user.clear(screen.getByLabelText("Number of similar reports"));
    await user.type(screen.getByLabelText("Number of similar reports"), "25");
    expect(
      fetchMock.mock.calls.some(([path]) => path.endsWith("top_n=25")),
    ).toBe(false);
    await user.click(screen.getByRole("button", { name: "Apply" }));
    await waitFor(() =>
      expect(
        fetchMock.mock.calls.some(([path]) => path.endsWith("top_n=25")),
      ).toBe(true),
    );
  });

  it("keeps maintenance collapsed and removes the outdated queued notice on refresh", async () => {
    open("/trends");
    await screen.findByText("Pattern maintenance");
    const summary = screen.getByText("Pattern maintenance");
    expect(summary.closest("details")?.open).toBe(false);
    await userEvent.click(summary);
    await userEvent.click(
      screen.getByRole("button", { name: "Rebuild patterns" }),
    );
    await screen.findByText(
      "Rebuild queued. Refresh to check completed patterns.",
    );
    await userEvent.click(screen.getByRole("button", { name: "Refresh" }));
    expect(
      screen.queryByText(
        "Rebuild queued. Refresh to check completed patterns.",
      ),
    ).toBeNull();
  });
});
