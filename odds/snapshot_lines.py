"""Append-only ESPN live-line snapshots. Safe to run from Task Scheduler."""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time
import urllib.error
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "odds"))

from pull_odds import fetch_scoreboard, parse_games, save_raw  # noqa: E402

SNAPSHOT_PATH = ROOT / "data" / "line_snapshots.csv"
LOG_PATH = ROOT / "data" / "line_snapshot_runs.log"
SNAPSHOT_HEADERS = [
    "timestamp",
    "season",
    "week",
    "game_id",
    "home_team",
    "away_team",
    "home_line",
]


def log(message: str) -> None:
    line = f"{datetime.now(timezone.utc).isoformat()}  {message}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def append_line_snapshots(season: int, week: int, games: list[dict]) -> int:
    """Append-only live-line log. One row per game that received a live home_line."""
    SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).isoformat()
    rows = []
    for game in games:
        line = game.get("vegas_spread")
        if line is None:
            continue
        rows.append(
            {
                "timestamp": ts,
                "season": season,
                "week": week,
                "game_id": game["game_id"],
                "home_team": game["home_team"],
                "away_team": game["away_team"],
                "home_line": line,
            }
        )
    if not rows:
        return 0
    frame = pd.DataFrame(rows, columns=SNAPSHOT_HEADERS)
    if SNAPSHOT_PATH.exists() and SNAPSHOT_PATH.stat().st_size > 0:
        frame.to_csv(SNAPSHOT_PATH, mode="a", header=False, index=False)
    else:
        frame.to_csv(SNAPSHOT_PATH, mode="w", header=True, index=False)
    return len(rows)


def run_once() -> int:
    raw, board, label = fetch_scoreboard()
    save_raw(raw, board, label)
    season = (board.get("season") or {}).get("year")
    week = (board.get("week") or {}).get("number")
    if season is None or week is None:
        raise RuntimeError("ESPN scoreboard missing season/week")
    season_i, week_i = int(season), int(week)
    odds = parse_games(board)
    games = []
    for rec in odds.itertuples(index=False):
        spread = getattr(rec, "market_spread", None)
        if spread is not None and pd.isna(spread):
            spread = None
        else:
            try:
                spread = float(spread) if spread is not None else None
            except (TypeError, ValueError):
                spread = None
        games.append(
            {
                "game_id": f"{season_i}_{week_i:02d}_{rec.away_team}_{rec.home_team}",
                "home_team": rec.home_team,
                "away_team": rec.away_team,
                "vegas_spread": spread,
            }
        )
    n = append_line_snapshots(season_i, week_i, games)
    log(
        f"ok  rows={n}  season={season_i}  week={week_i}  slate={label}  file={SNAPSHOT_PATH}"
    )
    return n


def run_once_safe() -> int:
    """One snapshot attempt. Logs failures and returns 0 instead of raising."""
    try:
        return run_once()
    except urllib.error.HTTPError as exc:
        log(f"error  HTTP {exc.code} {exc.reason}")
    except TimeoutError as exc:
        log(f"error  timeout  {exc}")
    except socket.timeout as exc:
        log(f"error  timeout  {exc}")
    except urllib.error.URLError as exc:
        log(f"error  URL error  {exc.reason}")
    except json.JSONDecodeError as exc:
        log(f"error  bad response (not JSON)  {exc}")
    except Exception as exc:  # noqa: BLE001 — scheduled runs must not die
        log(f"error  {type(exc).__name__}: {exc}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch ESPN lines and append snapshots")
    parser.add_argument(
        "--every-hours",
        type=float,
        default=0,
        help="If > 0, loop forever sleeping this many hours between runs",
    )
    args = parser.parse_args()
    if args.every_hours and args.every_hours > 0:
        while True:
            run_once_safe()
            time.sleep(args.every_hours * 3600)
    else:
        run_once_safe()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
