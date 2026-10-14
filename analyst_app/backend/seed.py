"""Authored synthetic fixtures only. Never reads any MIDAS source data."""

import argparse
from datetime import date, timedelta

from .config import Settings
from .contracts import EventInput
from .services import Services
from .storage import Database


def synthetic_events(as_of=None):
    today = as_of or date.today()
    week = today - timedelta(days=today.weekday() + 7)
    narratives = [
        (
            "BARCODE",
            "The barcode scanner failed during medication administration. A nurse caught an incorrect dose before the medication was given.",
            "Medical unit",
        ),
        (
            "FALL",
            "A patient had a fall while walking to the bathroom without assistance. The call bell was out of reach and the floor was wet.",
            "Rehabilitation",
        ),
        (
            "HANDOFF",
            "During the transfer handoff, the receiving team did not receive the updated care instructions. A callback clarified the missing information.",
            "Emergency department",
        ),
    ]
    result = []
    for group, narrative, location in narratives:
        for period in range(9):
            count = (8 if group == "BARCODE" else 3) if period == 8 else 1
            for j in range(count):
                result.append(
                    EventInput(
                        event_id=f"DEMO-{group}-{period:02}-{j:02}",
                        event_comments=narrative,
                        event_date=week - timedelta(weeks=8 - period) + timedelta(days=j % 6),
                        location_name=location,
                        encounter_service="Synthetic service",
                        facility="Demo campus",
                    )
                )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixture-mode",
        action="store_true",
        required=True,
        help="Required acknowledgment: classifier values are synthetic fixtures",
    )
    args = parser.parse_args()
    settings = Settings.from_env()
    settings.fixture_mode = args.fixture_mode
    db = Database(settings.database)
    if db.all("events"):
        raise SystemExit(
            "Refusing to mix demo fixtures with an existing database. Choose an empty MIDAS_DATABASE."
        )
    services = Services(settings, db)
    for event in synthetic_events():
        record = db.create_event(event)
        services.pending(record["id"])
        services.analyze(record["id"])
    services.rebuild()
    print(
        f"Created {len(db.all('events'))} authored synthetic reports. Rebuild: {services.rebuild_status['status']}; "
        f"themes: {len(db.all('themes'))}; embedding backend: {services.pattern.version}."
    )
    if services.startup_errors:
        print("Unavailable services:", ", ".join(services.startup_errors))


if __name__ == "__main__":
    main()
