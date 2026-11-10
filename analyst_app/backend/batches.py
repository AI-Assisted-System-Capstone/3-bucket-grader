"""Local Excel intake and worklist ranking for MIDAS exports."""

from datetime import date, datetime, time
from io import BytesIO
import re
from zipfile import ZipFile

from openpyxl import load_workbook
from pydantic import ValidationError

from .contracts import EventInput, FIELD_MAP, GRADES

MAX_UPLOAD_BYTES = 8 * 1024 * 1024
MAX_REPORTS = 500
REQUIRED = {"event_id": "Event No.", "event_date": "Event Date", "event_comments": "Event Comments"}


class ImportErrorDetail(ValueError):
    """Safe to display. Messages identify columns or row numbers, never values."""


def header_key(value):
    return re.sub(r"[^a-z0-9]", "", str(value or "").casefold())


HEADERS = {
    header_key(label): field
    for field, label in dict(FIELD_MAP, **REQUIRED, facility="Facility").items()
}
HEADERS.update({header_key(field): field for field in EventInput.model_fields})
HEADERS.update(
    {
        "eventno": "event_id",
        "narrativedescription": "event_comments",
        "location": "location_name",
        "facilityname": "facility",
        "meprescribedname": "prescribed_medication",
    }
)


def parse_date(value):
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value or "").strip()
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except ValueError:
        for fmt in ("%m/%d/%Y", "%m/%d/%Y %H:%M", "%m/%d/%Y %H:%M:%S", "%m/%d/%Y %I:%M:%S %p"):
            try:
                return datetime.strptime(text, fmt).date()
            except ValueError:
                continue
    raise ValueError("Invalid event date")


def read_workbook(content):
    try:
        with ZipFile(BytesIO(content)) as archive:
            if sum(entry.file_size for entry in archive.infolist()) > 64 * 1024 * 1024:
                raise ImportErrorDetail("The expanded workbook is too large.")
        workbook = load_workbook(
            BytesIO(content), read_only=True, data_only=False, keep_links=False
        )
        try:
            return read_sheet(workbook.worksheets[0])
        finally:
            workbook.close()
    except ImportErrorDetail:
        raise
    except Exception:
        raise ImportErrorDetail("Open a valid Excel .xlsx export and try again.") from None


def read_sheet(sheet):
    sheet.reset_dimensions()
    rows = iter(sheet.iter_rows())
    headers = next(rows, ())
    mapping = {}
    for index, cell in enumerate(headers):
        field = HEADERS.get(header_key(cell.value))
        if field:
            if field in mapping.values():
                raise ImportErrorDetail(f"More than one column maps to {field}.")
            mapping[index] = field
    missing = [label for field, label in REQUIRED.items() if field not in mapping.values()]
    if missing:
        raise ImportErrorDetail("Missing required first-sheet columns: " + ", ".join(missing) + ".")

    reports = []
    source_ids = set()
    for number, row in enumerate(rows, start=2):
        if (
            number > 5001
            or len(reports) >= MAX_REPORTS
            and any(cell.value is not None for cell in row)
        ):
            raise ImportErrorDetail(
                f"Split the export into files with at most {MAX_REPORTS} reports."
            )
        if not any(cell.value is not None for cell in row):
            continue
        fields = {}
        for index, field in mapping.items():
            if index >= len(row) or row[index].value is None:
                continue
            cell = row[index]
            if cell.data_type in ("f", "e"):
                raise ImportErrorDetail(
                    f"Row {number}: replace formulas or errors in {field} with values."
                )
            value = cell.value
            if isinstance(value, str) and not value.strip():
                continue
            if field in ("event_date", "date_received"):
                try:
                    fields[field] = parse_date(value)
                except ValueError:
                    raise ImportErrorDetail(
                        f"Row {number}: invalid {field.replace('_', ' ')}."
                    ) from None
            elif field == "event_time":
                if isinstance(value, datetime):
                    value = value.time()
                if not isinstance(value, time):
                    text = str(value).strip()
                    try:
                        value = time.fromisoformat(text.zfill(5) if len(text) == 4 else text)
                    except ValueError:
                        raise ImportErrorDetail(f"Row {number}: invalid event time.") from None
                fields[field] = value
            elif field == "age_at_encounter":
                fields[field] = value if str(value).strip() else None
            else:
                fields[field] = str(
                    int(value) if isinstance(value, float) and value.is_integer() else value
                )
        try:
            report = EventInput(**fields)
        except ValidationError as error:
            invalid = ", ".join(str(issue["loc"][0]) for issue in error.errors(include_input=False))
            raise ImportErrorDetail(f"Row {number}: check {invalid}.") from None
        if report.event_id in source_ids:
            raise ImportErrorDetail(f"Row {number}: repeated source event ID in this workbook.")
        source_ids.add(report.event_id)
        reports.append((number, report))
    if not reports:
        raise ImportErrorDetail("The first worksheet has no reports.")
    return reports


def review_tier(analysis):
    severity = (analysis or {}).get("severity")
    if not severity:
        return "unscored"
    if severity["priority_flag"]:
        return "immediate"
    if severity["grade"] in GRADES[GRADES.index("D") :]:
        return "secondary"
    return "routine"


TIERS = ("immediate", "secondary", "unscored", "routine")


def review_order(record):
    analysis = record.get("ai") or {}
    score = (analysis.get("severity") or {}).get("triage_score_raw", 0)
    return (TIERS.index(review_tier(analysis)), -score)
