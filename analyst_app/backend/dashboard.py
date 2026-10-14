"""Read-only dashboard projections. Analyst review never alters AI predictions."""

from datetime import date, timedelta

from .contracts import now


def event_rows(snapshot):
    ai = {r["event_id"]: r for r in snapshot["ai_results"]}
    reviewed = {r["event_id"] for r in snapshot["analyst_feedback"]}
    return sorted(
        [
            dict(
                **r,
                ai=ai.get(r["id"]),
                review_status="reviewed" if r["id"] in reviewed else "unreviewed",
            )
            for r in snapshot["events"]
        ],
        key=lambda r: (r["event_date"], r["created_at"], r["id"]),
        reverse=True,
    )


def is_priority(record):
    return bool(((record.get("ai") or {}).get("severity") or {}).get("priority_flag"))


def matches_filter(record, selected):
    if selected == "unreviewed":
        return record["review_status"] == "unreviewed"
    if selected == "priority":
        return record["review_status"] == "unreviewed" and is_priority(record)
    if selected == "reviewed":
        return record["review_status"] == "reviewed"
    if selected == "analysis_issues":
        return (record.get("ai") or {}).get("status") in ("partial", "failed")
    return True


def overview(snapshot, today=None):
    today = today or date.today()
    rows = event_rows(snapshot)
    waiting = [r for r in rows if matches_filter(r, "unreviewed")]
    # Stable sort preserves newest-first order within priority and ordinary groups.
    queue = sorted(waiting, key=lambda r: not is_priority(r))[:6]
    trends = {r["theme_id"]: r for r in snapshot["trend_results"]}
    last_complete_end = today - timedelta(days=today.weekday() + 1)
    patterns = []
    for theme in snapshot["themes"]:
        trend = trends.get(theme["theme_id"])
        patterns.append(
            dict(
                **{k: v for k, v in theme.items() if k != "centroid"},
                trend=trend,
                trend_stale=bool(trend and trend["period_end"] != last_complete_end.isoformat()),
            )
        )
    return dict(
        generated_at=now(),
        event_count=len(rows),
        reviewed_count=len(rows) - len(waiting),
        unreviewed_count=len(waiting),
        high_priority_count=sum(is_priority(r) for r in rows),
        priority_unreviewed_count=sum(is_priority(r) for r in waiting),
        analysis_issue_count=sum(matches_filter(r, "analysis_issues") for r in rows),
        pending_analysis_count=sum(
            (r.get("ai") or {}).get("status", "pending") == "pending" for r in rows
        ),
        emerging_pattern_count=sum(
            p["trend"] is not None and p["trend"]["status"] == "emerging" and not p["trend_stale"]
            for p in patterns
        ),
        review_queue=queue,
        recent_events=rows[:6],
        patterns=patterns,
    )
