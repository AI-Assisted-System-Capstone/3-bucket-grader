"""Fill a running analyst app with demo reports from the synthetic dataset.

Used by ../start_app.sh --demo. Builds three MIDAS-style Excel exports
(500 August, 500 September, 300 October 2024 reports; same random picks every
time) and uploads them through the app's own upload endpoint, oldest first, so
"similar reports" and the novelty flag have history to compare against. Patterns
are rebuilt after September (to set the novelty cut-off) and again at the end.

The exports contain every column of the dataset, including grades and review
notes; the upload step itself keeps only the intake fields. They are written to
analyst_app/local_state/demo_exports/, which git ignores. Does nothing if the app
already has uploads.
"""

import json
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

APP = Path(__file__).resolve().parent
sys.path.insert(0, str(APP.parent))
from data_pipeline import DATA_PATH  # noqa: E402

URL = "http://127.0.0.1:8000"
OUT = APP / "local_state" / "demo_exports"
XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
MONTHS = [("1_august", 8, 500), ("2_september", 9, 500), ("3_october", 10, 300)]


def call(path, data=None, content_type="application/json"):
    request = urllib.request.Request(URL + path, data=data, method="POST" if data else "GET")
    if data:
        request.add_header("content-type", content_type)
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


def wait(check, what, seconds=900):
    start = time.time()
    while time.time() - start < seconds:
        try:
            if check():
                return
        except OSError:
            pass  # server still starting
        time.sleep(2)
    sys.exit(f"Timed out waiting for {what}.")


def make_exports():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(DATA_PATH)
    dates = pd.to_datetime(df["Event Date"], errors="coerce")
    usable = df["event_comments"].fillna("").str.len() >= 10
    files = []
    for name, month, n in MONTHS:
        path = OUT / f"{name}.xlsx"
        if not path.exists():
            rows = df[(dates.dt.year == 2024) & (dates.dt.month == month) & usable]
            rows.sample(n, random_state=month).to_excel(path, index=False)
        files.append(path)
    return files


def upload(path):
    call("/api/batches", path.read_bytes(), XLSX)
    wait(lambda: call("/api/batches")[0]["status"] in ("complete", "failed"), f"{path.name} to be scored")
    print(f"  {path.name}: {call('/api/batches')[0]['status']}", flush=True)


def rebuild_patterns():
    call("/api/patterns/rebuild", b"{}")
    wait(lambda: call("/api/health")["rebuild"]["status"] in ("complete", "failed"), "patterns")


def main():
    wait(lambda: call("/api/health"), "the app to start", seconds=180)
    if call("/api/batches"):
        print("The app already has uploads; leaving them as they are.")
        return
    print("Building demo exports and uploading them (about 5 minutes)...", flush=True)
    files = make_exports()
    for i, path in enumerate(files):
        upload(path)
        if i == 1:
            rebuild_patterns()  # sets the novelty cut-off before October arrives
    rebuild_patterns()
    print("Demo reports loaded.")


if __name__ == "__main__":
    main()
