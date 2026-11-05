"""Isolated prediction-time levers: QB change, continuous rest, refined travel.

Each lever only adjusts predicted_diff when selected. Base EPA ratings stay
the recency + turnover standing model. Do not enable more than one at a time.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from rest_travel import rest_travel_features

# US timezones as hours behind ET. West (PT) vs East (ET) = 3.
TEAM_TZ_OFFSET: dict[str, int] = {
    "ATL": 0, "BAL": 0, "BUF": 0, "CAR": 0, "CIN": 0, "CLE": 0,
    "MIA": 0, "NE": 0, "NYG": 0, "NYJ": 0, "PHI": 0, "PIT": 0,
    "TB": 0, "WAS": 0,
    "CHI": 1, "DAL": 1, "DET": 1, "GB": 1, "HOU": 1, "IND": 1,
    "JAX": 1, "KC": 1, "MIN": 1, "NO": 1, "TEN": 1,
    "ARI": 2, "DEN": 2,
    "LA": 3, "LAC": 3, "LV": 3, "SEA": 3, "SF": 3,
}

STANDARD_REST_DAYS = 7.0
CONSERVATIVE_QB_GAP_EPA = 0.04  # ~1.2 pts at slope 31; TODO if data too thin
MIN_QB_CHANGE_GAMES = 8


def _mode_passer(passers: list[str]) -> str:
    counts = Counter(passers)
    best_n = max(counts.values())
    for passer in reversed(passers):
        if counts[passer] == best_n:
            return passer
    return passers[-1]


def qb_change_features(pbp: pd.DataFrame) -> pd.DataFrame:
    """Flag games whose primary passer differs from the mode of the prior 3 games."""
    needed = ["game_id", "week", "posteam", "pass_attempt", "passer_player_id", "game_date"]
    missing = [c for c in needed if c not in pbp.columns]
    if missing:
        raise ValueError(f"pbp missing columns: {missing}")

    att = pbp.loc[
        pd.to_numeric(pbp["pass_attempt"], errors="coerce").fillna(0) == 1,
        ["game_id", "week", "posteam", "passer_player_id", "game_date"],
    ].dropna(subset=["posteam", "passer_player_id", "game_id"])
    counts = (
        att.groupby(["game_id", "posteam", "passer_player_id"], as_index=False)
        .size()
        .rename(columns={"size": "n"})
    )
    idx = counts.groupby(["game_id", "posteam"])["n"].idxmax()
    primary = counts.loc[idx].rename(
        columns={"posteam": "team", "passer_player_id": "primary_passer"}
    )
    meta = (
        pbp.dropna(subset=["game_id"])
        .drop_duplicates("game_id")[["game_id", "week", "game_date", "home_team", "away_team"]]
        .copy()
    )
    meta["game_date"] = pd.to_datetime(meta["game_date"])
    primary = primary.merge(meta[["game_id", "week", "game_date"]], on="game_id", how="left")
    primary = primary.sort_values(["team", "game_date", "game_id"])

    flags = []
    for _, group in primary.groupby("team", sort=False):
        passers = group["primary_passer"].tolist()
        for i, passer in enumerate(passers):
            prior = passers[max(0, i - 3) : i]
            if not prior:
                flags.append(0)
            else:
                flags.append(int(passer != _mode_passer(prior)))
    primary = primary.copy()
    primary["qb_change"] = flags

    home = primary.rename(columns={"team": "home_team", "qb_change": "home_qb_change"})
    away = primary.rename(columns={"team": "away_team", "qb_change": "away_qb_change"})
    games = meta.merge(
        home[["game_id", "home_team", "home_qb_change"]],
        on=["game_id", "home_team"],
        how="left",
    ).merge(
        away[["game_id", "away_team", "away_qb_change"]],
        on=["game_id", "away_team"],
        how="left",
    )
    games["home_qb_change"] = games["home_qb_change"].fillna(0).astype(int)
    games["away_qb_change"] = games["away_qb_change"].fillna(0).astype(int)
    games["week"] = games["week"].astype(int)
    return games[
        [
            "game_id",
            "week",
            "home_team",
            "away_team",
            "home_qb_change",
            "away_qb_change",
        ]
    ]


def team_game_off_epa(pbp: pd.DataFrame) -> pd.DataFrame:
    plays = pbp.dropna(subset=["posteam", "epa", "game_id"]).copy()
    plays["week"] = plays["week"].astype(int)
    return (
        plays.groupby(["game_id", "posteam", "week"], as_index=False)["epa"]
        .mean()
        .rename(columns={"posteam": "team", "epa": "off_epa"})
    )


def backup_starter_gap_epa(
    off_epa: pd.DataFrame, qb_feat: pd.DataFrame, before_week: int
) -> tuple[float, str]:
    """Starter minus backup offensive EPA/play using only weeks < before_week.

    Returns (gap, note). Positive gap means backups were worse (reduce offense).
    """
    prior = qb_feat.loc[qb_feat["week"] < before_week]
    if prior.empty:
        return CONSERVATIVE_QB_GAP_EPA, "TODO conservative 0.04 (no prior weeks)"

    home = prior[["game_id", "week", "home_team", "home_qb_change"]].rename(
        columns={"home_team": "team", "home_qb_change": "qb_change"}
    )
    away = prior[["game_id", "week", "away_team", "away_qb_change"]].rename(
        columns={"away_team": "team", "away_qb_change": "qb_change"}
    )
    labeled = pd.concat([home, away], ignore_index=True)
    merged = off_epa.merge(labeled, on=["game_id", "team", "week"], how="inner")
    merged = merged.loc[merged["week"] < before_week]
    if merged.empty:
        return CONSERVATIVE_QB_GAP_EPA, "TODO conservative 0.04 (no matched EPA)"

    starter = merged.loc[merged["qb_change"] == 0, "off_epa"]
    backup = merged.loc[merged["qb_change"] == 1, "off_epa"]
    if len(backup) < MIN_QB_CHANGE_GAMES or starter.empty:
        return (
            CONSERVATIVE_QB_GAP_EPA,
            f"TODO conservative 0.04 (backup games={len(backup)} < {MIN_QB_CHANGE_GAMES})",
        )
    gap = float(starter.mean() - backup.mean())
    if gap <= 0:
        return (
            CONSERVATIVE_QB_GAP_EPA,
            f"TODO conservative 0.04 (data gap {gap:.4f} was non-positive)",
        )
    return gap, f"data gap={gap:.4f} EPA/play (n_backup={len(backup)}, n_starter={len(starter)})"


def qb_change_adjustment(current: pd.DataFrame, gap: float) -> pd.Series:
    """Reduce flagged team's offense: home change lowers home margin; away change raises it."""
    vals = gap * (
        current["away_qb_change"].astype(float) - current["home_qb_change"].astype(float)
    )
    return pd.Series(vals.to_numpy(dtype=float), index=current.index)


def apply_qb_change_layer(
    preds: pd.DataFrame,
    before_week: int,
    features: pd.DataFrame,
    off_epa: pd.DataFrame,
) -> tuple[pd.DataFrame, float, str]:
    """Walk-forward QB-change adjustment: gap from weeks < before_week only."""
    gap, note = backup_starter_gap_epa(off_epa, features, before_week)
    drop_cols = [c for c in ("home_qb_change", "away_qb_change") if c in preds.columns]
    out = preds.drop(columns=drop_cols).merge(
        features[["game_id", "home_qb_change", "away_qb_change"]],
        on="game_id",
        how="left",
    )
    for col in ("home_qb_change", "away_qb_change"):
        out[col] = out[col].fillna(0).astype(int)
    out["predicted_diff"] = out["predicted_diff"] + qb_change_adjustment(out, gap)
    return out, gap, note


def rest_continuous_features(pbp: pd.DataFrame) -> pd.DataFrame:
    raw = rest_travel_features(pbp)
    out = raw[["game_id", "home_rest_days", "away_rest_days"]].copy()
    out["home_rest_days"] = out["home_rest_days"].fillna(STANDARD_REST_DAYS)
    out["away_rest_days"] = out["away_rest_days"].fillna(STANDARD_REST_DAYS)
    out["home_rest_vs_std"] = out["home_rest_days"] - STANDARD_REST_DAYS
    out["away_rest_vs_std"] = out["away_rest_days"] - STANDARD_REST_DAYS
    return out


def travel_refined_features(pbp: pd.DataFrame) -> pd.DataFrame:
    raw = rest_travel_features(pbp)
    games = (
        pbp.dropna(subset=["game_id", "home_team", "away_team"])
        .drop_duplicates("game_id")[["game_id", "home_team", "away_team"]]
        .copy()
    )
    out = raw.merge(games, on="game_id", how="left")
    out["away_travel_kmi"] = out["away_travel_miles"].fillna(0.0) / 1000.0

    def tz(team) -> int:
        return int(TEAM_TZ_OFFSET.get(str(team), 0))

    out["away_tz_shift_3"] = (
        (out["away_team"].map(tz) - out["home_team"].map(tz)).abs() == 3
    ).astype(int)
    out["away_denver_altitude"] = (out["home_team"] == "DEN").astype(int)
    return out[
        ["game_id", "away_travel_kmi", "away_tz_shift_3", "away_denver_altitude"]
    ]


def ols_epa_adjustment(
    prior: pd.DataFrame, current: pd.DataFrame, feature_cols: list[str]
) -> pd.Series:
    """actual_diff ~ intercept + slope * predicted_diff + features; return EPA add-on."""
    zeros = pd.Series(0.0, index=current.index)
    needed = ["predicted_diff", "actual_diff", *feature_cols]
    if any(c not in prior.columns for c in needed) or len(prior) < 20:
        return zeros
    prior = prior.dropna(subset=needed)
    if len(prior) < 20:
        return zeros
    x = np.column_stack(
        [
            np.ones(len(prior)),
            prior["predicted_diff"].to_numpy(dtype=float),
            *[prior[c].to_numpy(dtype=float) for c in feature_cols],
        ]
    )
    y = prior["actual_diff"].to_numpy(dtype=float)
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    slope = float(coef[1])
    if abs(slope) < 1e-6:
        return zeros
    add = np.zeros(len(current), dtype=float)
    for i, col in enumerate(feature_cols):
        add += (float(coef[2 + i]) / slope) * current[col].to_numpy(dtype=float)
    return pd.Series(add, index=current.index)
