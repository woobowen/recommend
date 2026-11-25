"""Run the three scheduled read-only observations for this launch."""
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
start = datetime.fromisoformat(json.loads((HERE / "start-confirm.json").read_text())["time"])
for minutes, mode, label in ((10, "observe", "T10"), (20, "observe", "T20"), (30, "cutoff", "cutoff")):
    due = start + timedelta(minutes=minutes)
    print(datetime.now(timezone.utc).isoformat(), "waiting for", label, due.isoformat(), flush=True)
    while (remaining := (due - datetime.now(timezone.utc)).total_seconds()) > 0:
        time.sleep(min(remaining, 30))
    completed = subprocess.run([sys.executable, str(HERE / "collect.py"), mode, label], check=False)
    print(datetime.now(timezone.utc).isoformat(), label, "collection return code", completed.returncode, flush=True)
    folder = HERE / "observations" / label if mode == "observe" else HERE / label
    path = folder / "query.json"
    if path.exists() and json.loads(path.read_text())["status"] not in {"RUNNING", "ACCEPTED"}:
        print("terminal or supervisor anomaly, inspect existing query before any further checks", flush=True)
        break
