"""Append-only log of validation runs for comparing model layers."""

from datetime import date
from pathlib import Path

import pandas as pd

LOG_PATH = Path(__file__).resolve().parent / "metrics_log.csv"
HEADERS = [
    "date",
    "version_tag",
    "layers_active",
    "games",
    "r",
    "r_squared",
    "slope",
    "notes",
    "pick_accuracy_su",
    "pick_accuracy_ats",
]


def _ensure_schema() -> None:
    """Add any new columns to an existing log; leave old row values unchanged."""
    if not LOG_PATH.exists() or LOG_PATH.stat().st_size == 0:
        return
    existing = pd.read_csv(LOG_PATH)
    missing = [col for col in HEADERS if col not in existing.columns]
    if not missing:
        if list(existing.columns) != HEADERS:
            existing[HEADERS].to_csv(LOG_PATH, index=False)
        return
    for col in missing:
        existing[col] = pd.NA
    existing[HEADERS].to_csv(LOG_PATH, index=False)


def log_run(
    version_tag,
    layers_active,
    games,
    r,
    r_squared,
    slope,
    notes="",
    pick_accuracy_su=None,
    pick_accuracy_ats=None,
):
    """Append one validation result to metrics_log.csv, creating it if needed."""
    _ensure_schema()
    row = {
        "date": date.today().isoformat(),
        "version_tag": version_tag,
        "layers_active": layers_active,
        "games": games,
        "r": r,
        "r_squared": r_squared,
        "slope": slope,
        "notes": notes if notes is not None else "",
        "pick_accuracy_su": pick_accuracy_su,
        "pick_accuracy_ats": pick_accuracy_ats,
    }
    frame = pd.DataFrame([row], columns=HEADERS)
    if LOG_PATH.exists() and LOG_PATH.stat().st_size > 0:
        frame.to_csv(LOG_PATH, mode="a", header=False, index=False)
    else:
        frame.to_csv(LOG_PATH, mode="w", header=True, index=False)
    return row
