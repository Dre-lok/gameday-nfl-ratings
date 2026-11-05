"""Minimal FastAPI wrapper around the existing Gameday rating pipeline."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from scipy.stats import norm

GAME_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(GAME_ROOT / "models"))
sys.path.insert(0, str(GAME_ROOT / "odds"))

from pull_odds import fetch_scoreboard, parse_games, save_raw  # noqa: E402
from rating_v1 import usable_plays  # noqa: E402
from rating_v2 import PRIORS_PATH, blend_ratings  # noqa: E402
from snapshot_lines import append_line_snapshots  # noqa: E402
from levers import apply_qb_change_layer, qb_change_features, team_game_off_epa  # noqa: E402
from model_config import USE_QB_CHANGE  # noqa: E402
from validate_oos import predict_week, ratings_before_week  # noqa: E402

CALIBRATION_PATH = GAME_ROOT / "data" / "oos_calibration.json"
PBP_DIR = GAME_ROOT / "data" / "raw"
PBP_2026 = PBP_DIR / "pbp_2026.csv"
RELIABLE_READS_PATH = GAME_ROOT / "data" / "reliable_reads_log.csv"
RELIABLE_READS_HEADERS = [
    "date_logged",
    "season",
    "week",
    "game_id",
    "matchup",
    "model_p_home",
    "implied_side",
    "vegas_line",
]

app = FastAPI(title="gameday-dashboard")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def load_calibration() -> dict:
    return json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=8)
def load_season_pbp(season: int) -> pd.DataFrame:
    path = PBP_DIR / f"pbp_{season}.csv"
    if not path.exists():
        raise FileNotFoundError(str(path))
    return pd.read_csv(path, low_memory=False)


@lru_cache(maxsize=8)
def qb_change_tables(season: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    pbp = load_season_pbp(season)
    return qb_change_features(pbp), team_game_off_epa(pbp)


def predicted_margin_points(predicted_diff: float, cal: dict) -> float:
    return float(cal["intercept"] + cal["slope"] * predicted_diff)


def win_prob_home(margin_points: float, sigma: float) -> float:
    return float(1.0 - norm.cdf(0.5, loc=margin_points, scale=sigma))


def confidence_from_p_home(p_home: float) -> str:
    """Same rule as the UI Confidence column. Do not change the thresholds."""
    if p_home >= 0.6 or p_home <= 0.4:
        return "Reliable read"
    return "Toss-up zone"


def attach_vegas_implied(games: list[dict]) -> list[dict]:
    """Add vegas_implied_prob and divergence_label using the model's P(home) curve."""
    sigma = float(load_calibration()["residual_std"])
    for game in games:
        spread = game.get("vegas_spread")
        model_p = float(game["model_win_prob_home"])
        if spread is None:
            game["vegas_implied_prob"] = None
            game["divergence_label"] = None
            continue
        implied_margin = -float(spread)
        implied = round(win_prob_home(implied_margin, sigma), 3)
        gap_pp = abs(model_p - implied) * 100
        if gap_pp >= 15 and confidence_from_p_home(model_p) == "Reliable read":
            label = "Notable divergence"
        elif gap_pp >= 5:
            label = "Minor divergence"
        else:
            label = "Aligned"
        game["vegas_implied_prob"] = implied
        game["divergence_label"] = label
    return games


def vegas_home_spread(spread_line) -> float | None:
    """nflverse spread_line is positive when home is favored; UI uses Vegas home line."""
    if spread_line is None or (isinstance(spread_line, float) and pd.isna(spread_line)):
        return None
    try:
        return -float(spread_line)
    except (TypeError, ValueError):
        return None


def iso_or_none(value) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    return text if text else None


def games_played_2026() -> dict[str, int]:
    if not PBP_2026.exists():
        return {}
    pbp = pd.read_csv(PBP_2026, usecols=["posteam", "game_id"], low_memory=False)
    counts = (
        pbp.dropna(subset=["posteam", "game_id"])
        .drop_duplicates(["posteam", "game_id"])
        .groupby("posteam")
        .size()
        .to_dict()
    )
    return {str(k): int(v) for k, v in counts.items()}


def blended_nets() -> dict[str, float]:
    priors = pd.read_csv(PRIORS_PATH)
    prior = dict(zip(priors["team"], priors["prior_net_2026"]))
    played = games_played_2026()
    nets = {}
    for team in priors["team"]:
        n = int(played.get(team, 0))
        val = float(prior[team])
        nets[team] = blend_ratings(val, val, n) if n > 0 else val
    return nets


def ratings_lookup(ratings: pd.DataFrame) -> dict[str, dict]:
    by_team = {}
    for rec in ratings.itertuples(index=False):
        by_team[rec.team] = {
            "offense": round(float(rec.adj_offense), 4),
            "defense": round(float(rec.adj_defense), 4),
            "net": round(float(rec.adj_net), 4),
        }
    return by_team


def priors_lookup() -> dict[str, dict]:
    priors = pd.read_csv(PRIORS_PATH)
    by_team = {}
    for rec in priors.itertuples(index=False):
        by_team[rec.team] = {
            "offense": round(float(rec.off_2025), 4),
            "defense": round(float(rec.def_2025), 4),
            "net": round(float(rec.prior_net_2026), 4),
        }
    return by_team


def fetch_live_slate() -> dict | None:
    try:
        raw, board, label = fetch_scoreboard()
        save_raw(raw, board, label)
        season = (board.get("season") or {}).get("year")
        week = (board.get("week") or {}).get("number")
        if season is None or week is None:
            return None
        odds = parse_games(board)
        if odds.empty:
            return None
        return {
            "season": int(season),
            "week": int(week),
            "label": label,
            "odds": odds,
        }
    except Exception as exc:
        print(f"live ESPN fetch failed: {exc}")
        return None


def game_datetime_map(pbp: pd.DataFrame, game_ids) -> dict[str, str | None]:
    cols = [c for c in ("game_id", "time_of_day", "start_time", "game_date") if c in pbp.columns]
    if "game_id" not in cols:
        return {}
    sub = pbp.loc[pbp["game_id"].isin(game_ids), cols]
    out: dict[str, str | None] = {}
    for gid, grp in sub.groupby("game_id"):
        val = None
        for col in ("time_of_day", "start_time", "game_date"):
            if col not in grp.columns:
                continue
            series = grp[col].dropna()
            if series.empty:
                continue
            text = str(series.iloc[0]).strip()
            if text and text.lower() != "nan":
                val = text
                break
        out[str(gid)] = val
    return out


def overlay_live_lines(games: list[dict], slate: dict, season: int, week: int) -> list[dict]:
    """Replace stored close with live ESPN home line for the current slate only."""
    by_matchup = {}
    for rec in slate["odds"].itertuples(index=False):
        spread = getattr(rec, "market_spread", None)
        if spread is not None and pd.isna(spread):
            spread = None
        else:
            try:
                spread = float(spread) if spread is not None else None
            except (TypeError, ValueError):
                spread = None
        kickoff = getattr(rec, "kickoff", None)
        by_matchup[(rec.home_team, rec.away_team)] = (spread, iso_or_none(kickoff))

    live_games = []
    for game in games:
        key = (game["home_team"], game["away_team"])
        if key not in by_matchup:
            continue
        spread, kickoff = by_matchup[key]
        game["vegas_spread"] = spread
        game["vegas_source"] = "live_espn"
        if kickoff:
            game["game_datetime"] = kickoff
            game["game_date"] = kickoff[:10]
        live_games.append(game)
    append_line_snapshots(season, week, live_games)
    return games


def week_from_pbp(season: int, week: int) -> list[dict]:
    pbp = load_season_pbp(season)
    pbp = pbp.copy()
    pbp["week"] = pbp["week"].astype(int)
    games = (
        pbp.dropna(subset=["game_id", "home_team", "away_team"])
        .drop_duplicates("game_id")
        .copy()
    )
    games["week"] = games["week"].astype(int)
    week_games = games.loc[games["week"] == week]
    if week_games.empty:
        return []

    plays = usable_plays(pbp).copy()
    plays["week"] = plays["week"].astype(int)
    prior = plays.loc[plays["week"] < week]
    if prior.empty:
        raise HTTPException(
            status_code=400,
            detail="No prior weeks to build ratings. Use week 5+ for in-season walk-forward.",
        )

    ratings, hfa = ratings_before_week(plays, week)
    units = ratings_lookup(ratings)
    needed = ["game_id", "week", "home_team", "away_team", "home_score", "away_score"]
    slim = week_games[needed].copy()
    slim["actual_diff"] = slim["home_score"] - slim["away_score"]
    preds = predict_week(slim, ratings, hfa)
    if USE_QB_CHANGE:
        features, off_epa = qb_change_tables(season)
        preds, _, _ = apply_qb_change_layer(preds, week, features, off_epa)

    meta = week_games.set_index("game_id")
    datetimes = game_datetime_map(pbp, week_games["game_id"])
    cal = load_calibration()
    sigma = float(cal["residual_std"])
    rows = []
    for rec in preds.itertuples(index=False):
        margin = predicted_margin_points(float(rec.predicted_diff), cal)
        game_date = meta.loc[rec.game_id, "game_date"] if "game_date" in meta.columns else None
        spread_line = meta.loc[rec.game_id, "spread_line"] if "spread_line" in meta.columns else None
        game_date = None if pd.isna(game_date) else str(game_date)[:10]
        datetime_full = iso_or_none(datetimes.get(rec.game_id)) or game_date
        home = units.get(rec.home_team, {})
        away = units.get(rec.away_team, {})
        rows.append(
            {
                "game_id": rec.game_id,
                "game_date": game_date,
                "game_datetime": datetime_full,
                "home_team": rec.home_team,
                "away_team": rec.away_team,
                "model_predicted_margin": round(margin, 1),
                "model_win_prob_home": round(win_prob_home(margin, sigma), 3),
                "vegas_spread": vegas_home_spread(spread_line),
                "vegas_source": "historical_close",
                "home_offense": home.get("offense"),
                "home_defense": home.get("defense"),
                "away_offense": away.get("offense"),
                "away_defense": away.get("defense"),
            }
        )
    rows.sort(key=lambda r: (r["game_date"] or "", r["home_team"]))
    return rows


def week_from_2026_priors(week: int, live: dict | None) -> list[dict]:
    """Cold-start 2026: 2026 priors + live ESPN slate when it matches this week."""
    if not PRIORS_PATH.exists():
        raise HTTPException(status_code=404, detail="2026 priors not found")
    if live is None or int(live["week"]) != week:
        return []

    units = priors_lookup()
    nets = blended_nets()
    cal = load_calibration()
    hfa = float(cal.get("hfa_epa", 0.0))
    sigma = float(cal["residual_std"])
    rows = []
    for rec in live["odds"].itertuples(index=False):
        home, away = rec.home_team, rec.away_team
        if home not in nets or away not in nets:
            continue
        pred_diff = nets[home] - nets[away] + hfa
        margin = predicted_margin_points(pred_diff, cal)
        kickoff = iso_or_none(getattr(rec, "kickoff", None))
        spread = getattr(rec, "market_spread", None)
        if spread is not None and pd.isna(spread):
            spread = None
        else:
            try:
                spread = float(spread) if spread is not None else None
            except (TypeError, ValueError):
                spread = None
        home_u = units.get(home, {})
        away_u = units.get(away, {})
        rows.append(
            {
                "game_id": f"2026_{week:02d}_{away}_{home}",
                "game_date": kickoff[:10] if kickoff else None,
                "game_datetime": kickoff,
                "home_team": home,
                "away_team": away,
                "model_predicted_margin": round(margin, 1),
                "model_win_prob_home": round(win_prob_home(margin, sigma), 3),
                "vegas_spread": spread,
                "vegas_source": "live_espn",
                "home_offense": home_u.get("offense"),
                "home_defense": home_u.get("defense"),
                "away_offense": away_u.get("offense"),
                "away_defense": away_u.get("defense"),
            }
        )
    rows.sort(key=lambda r: (r["game_date"] or "", r["home_team"]))
    append_line_snapshots(2026, week, rows)
    return rows


def load_week_games(season: int, week: int) -> list[dict]:
    if week < 1 or week > 22:
        raise HTTPException(status_code=400, detail="week must be 1-22")
    try:
        if season == 2026 and not PBP_2026.exists():
            live = fetch_live_slate()
            if live is None:
                raise HTTPException(
                    status_code=502,
                    detail="Could not fetch live ESPN odds for the current slate.",
                )
            games = week_from_2026_priors(week, live)
        else:
            games = week_from_pbp(season, week)
            if season >= 2026:
                live = fetch_live_slate()
                if live and live["season"] == season and live["week"] == week:
                    games = overlay_live_lines(games, live, season, week)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"No play-by-play file for {season}") from None
    return attach_vegas_implied(games)


def implied_side(p_home: float) -> str:
    return "home" if p_home >= 0.5 else "away"


def read_reliable_log() -> pd.DataFrame:
    if not RELIABLE_READS_PATH.exists() or RELIABLE_READS_PATH.stat().st_size == 0:
        return pd.DataFrame(columns=RELIABLE_READS_HEADERS)
    return pd.read_csv(RELIABLE_READS_PATH)


def final_scores_for_season(season: int) -> dict[str, tuple[float, float]]:
    """Final home/away scores from existing pbp. Empty if that season has no file."""
    path = PBP_DIR / f"pbp_{season}.csv"
    if not path.exists():
        return {}
    pbp = pd.read_csv(path, usecols=["game_id", "home_score", "away_score"], low_memory=False)
    pbp = pbp.dropna(subset=["game_id", "home_score", "away_score"])
    if pbp.empty:
        return {}
    last = pbp.groupby("game_id", sort=False).last()
    out = {}
    for gid, rec in last.iterrows():
        out[str(gid)] = (float(rec.home_score), float(rec.away_score))
    return out


def grade_logged_row(row: dict, scores: dict[str, tuple[float, float]]) -> dict:
    gid = str(row["game_id"])
    if gid not in scores:
        row["status"] = "Pending"
        row["home_score"] = None
        row["away_score"] = None
        return row
    home_s, away_s = scores[gid]
    row["home_score"] = home_s
    row["away_score"] = away_s
    if home_s == away_s:
        row["status"] = "Miss"
        return row
    winner = "home" if home_s > away_s else "away"
    row["status"] = "Hit" if winner == row["implied_side"] else "Miss"
    return row


@app.get("/api/week/{season}/{week}")
def get_week(season: int, week: int) -> dict:
    games = load_week_games(season, week)
    return {"season": season, "week": week, "games": games}


@app.get("/api/reliable-reads")
def list_reliable_reads() -> dict:
    log = read_reliable_log()
    by_season: dict[int, dict] = {}
    rows = []
    for rec in log.to_dict(orient="records"):
        season = int(rec["season"])
        if season not in by_season:
            by_season[season] = final_scores_for_season(season)
        rows.append(grade_logged_row(rec, by_season[season]))
    hits = sum(1 for r in rows if r["status"] == "Hit")
    misses = sum(1 for r in rows if r["status"] == "Miss")
    pending = sum(1 for r in rows if r["status"] == "Pending")
    graded = hits + misses
    return {
        "picks": rows,
        "record": {
            "hits": hits,
            "misses": misses,
            "pending": pending,
            "graded": graded,
            "win_pct": (hits / graded) if graded else None,
        },
    }


@app.post("/api/reliable-reads/log/{season}/{week}")
def log_reliable_reads(season: int, week: int) -> dict:
    games = load_week_games(season, week)
    reliable = [
        g for g in games if confidence_from_p_home(float(g["model_win_prob_home"])) == "Reliable read"
    ]
    existing = read_reliable_log()
    known = set(existing["game_id"].astype(str)) if not existing.empty else set()
    ts = datetime.now(timezone.utc).isoformat()
    new_rows = []
    skipped = 0
    for game in reliable:
        gid = str(game["game_id"])
        if gid in known:
            skipped += 1
            continue
        known.add(gid)
        p_home = float(game["model_win_prob_home"])
        spread = game.get("vegas_spread")
        new_rows.append(
            {
                "date_logged": ts,
                "season": season,
                "week": week,
                "game_id": gid,
                "matchup": f"{game['away_team']} @ {game['home_team']}",
                "model_p_home": p_home,
                "implied_side": implied_side(p_home),
                "vegas_line": spread if spread is not None else "",
            }
        )
    if new_rows:
        RELIABLE_READS_PATH.parent.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame(new_rows, columns=RELIABLE_READS_HEADERS)
        if RELIABLE_READS_PATH.exists() and RELIABLE_READS_PATH.stat().st_size > 0:
            frame.to_csv(RELIABLE_READS_PATH, mode="a", header=False, index=False)
        else:
            frame.to_csv(RELIABLE_READS_PATH, mode="w", header=True, index=False)
    listing = list_reliable_reads()
    listing["appended"] = len(new_rows)
    listing["skipped"] = skipped
    listing["reliable_this_week"] = len(reliable)
    return listing
