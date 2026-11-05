"""Rest days and travel distance as a togglable prediction-time layer.

Does not change base EPA ratings. When the flag is on, walk-forward fits a
small additive adjustment from prior weeks' short-rest / long-travel dummies
and adds it to predicted_diff. When the flag is off, predictions are unchanged.
"""

from __future__ import annotations

from math import asin, cos, radians, sin, sqrt

import numpy as np
import pandas as pd

# nflverse team abbrev -> home stadium lat/lon. NYG/NYJ share MetLife; LA/LAC share SoFi.
STADIUM_COORDS: dict[str, tuple[float, float]] = {
    "ARI": (33.5276, -112.2626),
    "ATL": (33.7553, -84.4006),
    "BAL": (39.2780, -76.6227),
    "BUF": (42.7738, -78.7869),
    "CAR": (35.2258, -80.8528),
    "CHI": (41.8623, -87.6167),
    "CIN": (39.0954, -84.5160),
    "CLE": (41.5061, -81.6995),
    "DAL": (32.7478, -97.0928),
    "DEN": (39.7439, -105.0201),
    "DET": (42.3400, -83.0456),
    "GB": (44.5013, -88.0622),
    "HOU": (29.6847, -95.4107),
    "IND": (39.7601, -86.1639),
    "JAX": (30.3239, -81.6373),
    "KC": (39.0489, -94.4839),
    "LA": (33.9535, -118.3390),
    "LAC": (33.9535, -118.3390),
    "LV": (36.0909, -115.1833),
    "MIA": (25.9580, -80.2389),
    "MIN": (44.9738, -93.2577),
    "NE": (42.0909, -71.2643),
    "NO": (29.9511, -90.0812),
    "NYG": (40.8128, -74.0742),
    "NYJ": (40.8128, -74.0742),
    "PHI": (39.9008, -75.1675),
    "PIT": (40.4468, -80.0158),
    "SEA": (47.5952, -122.3316),
    "SF": (37.4033, -121.9694),
    "TB": (27.9759, -82.5033),
    "TEN": (36.1665, -86.7713),
    "WAS": (38.9078, -76.8645),
}

EARTH_MILES = 3958.8
SHORT_REST_DAYS = 4
LONG_TRAVEL_MILES = 1500.0


def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    rlat1, rlon1, rlat2, rlon2 = map(radians, (lat1, lon1, lat2, lon2))
    dlat = rlat2 - rlat1
    dlon = rlon2 - rlon1
    a = sin(dlat / 2) ** 2 + cos(rlat1) * cos(rlat2) * sin(dlon / 2) ** 2
    return 2 * EARTH_MILES * asin(sqrt(a))


def _coords(team: str) -> tuple[float, float] | None:
    return STADIUM_COORDS.get(str(team))


def rest_travel_features(pbp: pd.DataFrame) -> pd.DataFrame:
    """One row per game: rest days, short-rest flags, away travel miles."""
    needed = ["game_id", "week", "home_team", "away_team", "game_date"]
    missing = [c for c in needed if c not in pbp.columns]
    if missing:
        raise ValueError(f"pbp missing columns: {missing}")

    games = (
        pbp.dropna(subset=["game_id", "home_team", "away_team", "game_date"])
        .drop_duplicates("game_id")[needed]
        .copy()
    )
    games["game_date"] = pd.to_datetime(games["game_date"])
    games["week"] = games["week"].astype(int)

    appearances = []
    for rec in games.itertuples(index=False):
        appearances.append(
            {
                "game_id": rec.game_id,
                "team": rec.home_team,
                "game_date": rec.game_date,
                "venue": rec.home_team,
            }
        )
        appearances.append(
            {
                "game_id": rec.game_id,
                "team": rec.away_team,
                "game_date": rec.game_date,
                "venue": rec.home_team,
            }
        )
    long = pd.DataFrame(appearances).sort_values(["team", "game_date", "game_id"])
    long["prev_date"] = long.groupby("team")["game_date"].shift(1)
    long["prev_venue"] = long.groupby("team")["venue"].shift(1)
    long["rest_days"] = (long["game_date"] - long["prev_date"]).dt.days

    def _travel(row) -> float:
        dest = _coords(row.venue)
        if dest is None:
            return 0.0
        if pd.isna(row.prev_venue):
            origin = _coords(row.team)
        else:
            origin = _coords(row.prev_venue)
        if origin is None:
            return 0.0
        return haversine_miles(origin[0], origin[1], dest[0], dest[1])

    long["travel_miles"] = long.apply(_travel, axis=1)
    team_game = long.set_index(["game_id", "team"])

    rows = []
    for rec in games.itertuples(index=False):
        home = team_game.loc[(rec.game_id, rec.home_team)]
        away = team_game.loc[(rec.game_id, rec.away_team)]
        home_rest = home["rest_days"]
        away_rest = away["rest_days"]
        rows.append(
            {
                "game_id": rec.game_id,
                "home_rest_days": home_rest,
                "away_rest_days": away_rest,
                "home_short_rest": int(pd.notna(home_rest) and home_rest <= SHORT_REST_DAYS),
                "away_short_rest": int(pd.notna(away_rest) and away_rest <= SHORT_REST_DAYS),
                "away_travel_miles": float(away["travel_miles"]),
                "away_long_travel": int(float(away["travel_miles"]) >= LONG_TRAVEL_MILES),
            }
        )
    return pd.DataFrame(rows)


def merge_features(games: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    out = games.merge(features, on="game_id", how="left")
    for col in ("home_short_rest", "away_short_rest", "away_long_travel"):
        out[col] = out[col].fillna(0).astype(int)
    out["away_travel_miles"] = out["away_travel_miles"].fillna(0.0)
    return out


def adjustment_from_prior(prior: pd.DataFrame, current: pd.DataFrame) -> pd.Series:
    """Fit dummy effects on prior games; return EPA-scale add-on for current games.

    OLS: actual_diff ~ intercept + slope * predicted_diff + short-rest/travel dummies.
    Dummy coefficients are in points; divide by slope to stay in predicted_diff units.
    """
    zeros = pd.Series(0.0, index=current.index)
    needed = [
        "predicted_diff",
        "actual_diff",
        "home_short_rest",
        "away_short_rest",
        "away_long_travel",
    ]
    if any(c not in prior.columns for c in needed) or len(prior) < 20:
        return zeros
    prior = prior.dropna(subset=needed)
    if len(prior) < 20:
        return zeros
    if (
        int(prior["home_short_rest"].sum())
        + int(prior["away_short_rest"].sum())
        + int(prior["away_long_travel"].sum())
        < 4
    ):
        return zeros

    x = np.column_stack(
        [
            np.ones(len(prior)),
            prior["predicted_diff"].to_numpy(dtype=float),
            prior["home_short_rest"].to_numpy(dtype=float),
            prior["away_short_rest"].to_numpy(dtype=float),
            prior["away_long_travel"].to_numpy(dtype=float),
        ]
    )
    y = prior["actual_diff"].to_numpy(dtype=float)
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    slope = float(coef[1])
    if abs(slope) < 1e-6:
        return zeros
    add = (
        (float(coef[2]) / slope) * current["home_short_rest"].to_numpy(dtype=float)
        + (float(coef[3]) / slope) * current["away_short_rest"].to_numpy(dtype=float)
        + (float(coef[4]) / slope) * current["away_long_travel"].to_numpy(dtype=float)
    )
    return pd.Series(add, index=current.index)
