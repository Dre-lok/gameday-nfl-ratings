"""Key starter-change lever: WR1, CB1/top DB, and OL from official injury reports.

Flag is pre-game: last week's snap-count starter (WR1 / CB1 / starting OL) is
listed Out or Doubtful on this week's final nflverse injury report.
QB is handled separately by the qb_change lever and is ignored here.
"""

from __future__ import annotations

import io
import os
import ssl
import urllib.request
from pathlib import Path

import pandas as pd

from rating_v1 import ROOT

try:
    import certifi

    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
    ssl._create_default_https_context = lambda: ssl.create_default_context(
        cafile=certifi.where()
    )
except ImportError:
    certifi = None  # type: ignore[assignment]

SNAP_PATH = ROOT / "data" / "raw" / "snap_counts_2025.csv"
INJURY_PATH = ROOT / "data" / "raw" / "injuries_2023_2025.csv"
INJURY_SEASONS = (2023, 2024, 2025)
NFLVERSE_INJURIES = (
    "https://github.com/nflverse/nflverse-data/releases/download/injuries/"
    "injuries_{year}.parquet"
)

OL_POS = {"C", "G", "T", "OL"}
WR_POS = {"WR"}
CB_POS = {"CB"}
DB_POS = {"DB"}
OUT_STATUSES = {"out", "doubtful"}
CONSERVATIVE_STARTER_GAP_EPA = 0.03  # TODO if data gap is thin or non-positive
MIN_CHANGE_GAMES = 8


def _ssl_context() -> ssl.SSLContext:
    if certifi is not None:
        return ssl.create_default_context(cafile=certifi.where())
    return ssl.create_default_context()


def _norm_name(series: pd.Series) -> pd.Series:
    out = series.fillna("").astype(str).str.lower()
    out = out.str.replace(r"[^a-z ]", "", regex=True)
    out = out.str.replace(r"\s+", " ", regex=True).str.strip()
    out = out.str.replace(r"\b(jr|sr|ii|iii|iv|v)\b", "", regex=True)
    return out.str.replace(r"\s+", " ", regex=True).str.strip()


def load_snap_counts() -> pd.DataFrame:
    if SNAP_PATH.exists() and SNAP_PATH.stat().st_size > 0:
        return pd.read_csv(SNAP_PATH)
    import nfl_data_py as nfl

    snaps = nfl.import_snap_counts([2025])
    SNAP_PATH.parent.mkdir(parents=True, exist_ok=True)
    snaps.to_csv(SNAP_PATH, index=False)
    return snaps


def _download_injuries_year(year: int) -> pd.DataFrame:
    url = NFLVERSE_INJURIES.format(year=year)
    req = urllib.request.Request(url, headers={"User-Agent": "gameday-injuries-pull"})
    with urllib.request.urlopen(req, context=_ssl_context()) as resp:
        raw = resp.read()
    return pd.read_parquet(io.BytesIO(raw))


def load_injuries(years: tuple[int, ...] = INJURY_SEASONS) -> pd.DataFrame:
    """Official weekly injury reports for the walk-forward pool seasons."""
    years = tuple(years)
    if INJURY_PATH.exists() and INJURY_PATH.stat().st_size > 0:
        cached = pd.read_csv(INJURY_PATH)
        have = set(pd.to_numeric(cached["season"], errors="coerce").dropna().astype(int))
        if set(years).issubset(have):
            return cached.loc[cached["season"].isin(years)].copy()
    frames = []
    try:
        for year in years:
            frames.append(_download_injuries_year(year))
        injuries = pd.concat(frames, ignore_index=True)
    except Exception:
        import nfl_data_py as nfl

        injuries = nfl.import_injuries(list(years))
    INJURY_PATH.parent.mkdir(parents=True, exist_ok=True)
    injuries.to_csv(INJURY_PATH, index=False)
    return injuries


def team_game_net_epa(pbp: pd.DataFrame) -> pd.DataFrame:
    plays = pbp.dropna(subset=["posteam", "defteam", "epa", "game_id"]).copy()
    plays["week"] = plays["week"].astype(int)
    off = (
        plays.groupby(["game_id", "posteam", "week"], as_index=False)["epa"]
        .mean()
        .rename(columns={"posteam": "team", "epa": "off_epa"})
    )
    allowed = (
        plays.groupby(["game_id", "defteam", "week"], as_index=False)["epa"]
        .mean()
        .rename(columns={"defteam": "team", "epa": "epa_allowed"})
    )
    net = off.merge(allowed, on=["game_id", "team", "week"], how="inner")
    net["net_epa"] = net["off_epa"] - net["epa_allowed"]
    return net


def _starters_for_group(group: pd.DataFrame) -> dict:
    ol = group.loc[group["position"].isin(OL_POS)].sort_values(
        ["offense_snaps", "offense_pct"], ascending=False
    )
    wr = group.loc[group["position"].isin(WR_POS)].sort_values(
        ["offense_snaps", "offense_pct"], ascending=False
    )
    cb = group.loc[group["position"].isin(CB_POS)].sort_values(
        ["defense_snaps", "defense_pct"], ascending=False
    )
    if cb.empty:
        cb = group.loc[group["position"].isin(DB_POS)].sort_values(
            ["defense_snaps", "defense_pct"], ascending=False
        )
    ol_names = tuple(n for n in ol["player_norm"].head(5).tolist() if n)
    wr1 = wr["player_norm"].iloc[0] if not wr.empty else None
    cb1 = cb["player_norm"].iloc[0] if not cb.empty else None
    cb_listed = group.loc[group["position"].isin(CB_POS)]
    return {
        "ol": ol_names,
        "wr1": wr1 if wr1 else None,
        "cb1": cb1 if cb1 else None,
        "cb_from_db": bool(cb_listed.empty and cb1 is not None),
    }


def _injury_lookup(injuries: pd.DataFrame) -> dict[tuple, str]:
    """(season, team, week, norm_name) -> worst report_status on that week."""
    inj = injuries.copy()
    inj["season"] = pd.to_numeric(inj["season"], errors="coerce").astype("Int64")
    inj["week"] = pd.to_numeric(inj["week"], errors="coerce").astype("Int64")
    inj["player_norm"] = _norm_name(inj["full_name"])
    inj["status_norm"] = inj["report_status"].fillna("").astype(str).str.strip().str.lower()
    inj = inj.loc[inj["player_norm"].ne("") & inj["season"].notna() & inj["week"].notna()]
    rank = {"out": 3, "doubtful": 2, "questionable": 1}
    inj["rank"] = inj["status_norm"].map(rank).fillna(0)
    inj = inj.sort_values("rank")
    lookup: dict[tuple, str] = {}
    ranks: dict[tuple, float] = {}
    for rec in inj.itertuples(index=False):
        keys = [(int(rec.season), rec.team, int(rec.week), rec.player_norm)]
        parts = rec.player_norm.split()
        if len(parts) >= 2:
            keys.append((int(rec.season), rec.team, int(rec.week), f"{parts[0]} {parts[-1]}"))
        for key in keys:
            prev_rank = ranks.get(key, -1)
            if rec.rank >= prev_rank:
                lookup[key] = rec.status_norm
                ranks[key] = rec.rank
    return lookup


def _is_out_or_doubtful(lookup: dict[tuple, str], season: int, team: str, week: int, name: str | None) -> int:
    if not name:
        return 0
    keys = [
        (season, team, week, name),
    ]
    parts = name.split()
    if len(parts) >= 2:
        keys.append((season, team, week, f"{parts[0]} {parts[-1]}"))
    for key in keys:
        if lookup.get(key, "") in OUT_STATUSES:
            return 1
    return 0


def starter_change_features(
    pbp: pd.DataFrame,
    snaps: pd.DataFrame | None = None,
    injuries: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Flag team-games whose WR1, CB1, or a starting OL is Out/Doubtful."""
    if snaps is None:
        snaps = load_snap_counts()
    if injuries is None:
        injuries = load_injuries()
    snaps = snaps.copy()
    snaps["week"] = snaps["week"].astype(int)
    if "season" not in snaps.columns:
        snaps["season"] = pd.to_numeric(snaps["game_id"].astype(str).str.slice(0, 4), errors="coerce")
    snaps["season"] = snaps["season"].fillna(2025).astype(int)
    snaps["player_norm"] = _norm_name(snaps["player"] if "player" in snaps.columns else snaps.get("player_name", pd.Series("", index=snaps.index)))
    lookup = _injury_lookup(injuries)

    rows = []
    n_cb_db_fallback = 0
    for (game_id, team), group in snaps.groupby(["game_id", "team"], sort=False):
        starters = _starters_for_group(group)
        if starters["cb_from_db"]:
            n_cb_db_fallback += 1
        rows.append(
            {
                "game_id": game_id,
                "team": team,
                "week": int(group["week"].iloc[0]),
                "season": int(group["season"].iloc[0]),
                "ol": starters["ol"],
                "wr1": starters["wr1"],
                "cb1": starters["cb1"],
            }
        )
    long = pd.DataFrame(rows).sort_values(["team", "season", "week", "game_id"])

    flags = []
    for _, group in long.groupby(["team", "season"], sort=False):
        prev_ol = prev_wr = prev_cb = None
        for rec in group.itertuples(index=False):
            if prev_ol is None and prev_wr is None and prev_cb is None:
                ol_ch = wr_ch = cb_ch = 0
            else:
                ol_ch = 0
                for ol_name in prev_ol or ():
                    ol_ch = max(
                        ol_ch,
                        _is_out_or_doubtful(lookup, rec.season, rec.team, rec.week, ol_name),
                    )
                wr_ch = _is_out_or_doubtful(lookup, rec.season, rec.team, rec.week, prev_wr)
                cb_ch = _is_out_or_doubtful(lookup, rec.season, rec.team, rec.week, prev_cb)
            flags.append(
                {
                    "game_id": rec.game_id,
                    "team": rec.team,
                    "week": rec.week,
                    "ol_change": ol_ch,
                    "wr_change": wr_ch,
                    "cb_change": cb_ch,
                    "starter_change": int(bool(ol_ch or wr_ch or cb_ch)),
                }
            )
            prev_ol, prev_wr, prev_cb = rec.ol, rec.wr1, rec.cb1
    flagged = pd.DataFrame(flags)

    meta = (
        pbp.dropna(subset=["game_id", "home_team", "away_team"])
        .drop_duplicates("game_id")[["game_id", "week", "home_team", "away_team"]]
        .copy()
    )
    meta["week"] = meta["week"].astype(int)
    home_detail = flagged.rename(
        columns={
            "team": "home_team",
            "ol_change": "home_ol_change",
            "wr_change": "home_wr_change",
            "cb_change": "home_cb_change",
            "starter_change": "home_starter_change",
        }
    )
    away_detail = flagged.rename(
        columns={
            "team": "away_team",
            "ol_change": "away_ol_change",
            "wr_change": "away_wr_change",
            "cb_change": "away_cb_change",
            "starter_change": "away_starter_change",
        }
    )
    games = meta.merge(
        home_detail[
            [
                "game_id",
                "home_team",
                "home_starter_change",
                "home_ol_change",
                "home_wr_change",
                "home_cb_change",
            ]
        ],
        on=["game_id", "home_team"],
        how="left",
    ).merge(
        away_detail[
            [
                "game_id",
                "away_team",
                "away_starter_change",
                "away_ol_change",
                "away_wr_change",
                "away_cb_change",
            ]
        ],
        on=["game_id", "away_team"],
        how="left",
    )
    for col in (
        "home_starter_change",
        "away_starter_change",
        "home_ol_change",
        "away_ol_change",
        "home_wr_change",
        "away_wr_change",
        "home_cb_change",
        "away_cb_change",
    ):
        games[col] = games[col].fillna(0).astype(int)
    games.attrs["cb_db_fallback_team_games"] = int(n_cb_db_fallback)
    games.attrs["injury_seasons"] = sorted(
        {int(y) for y in pd.to_numeric(injuries["season"], errors="coerce").dropna().unique()}
    )
    return games


def starter_change_gap_epa(
    net_epa: pd.DataFrame, feat: pd.DataFrame, before_week: int
) -> tuple[float, str]:
    prior = feat.loc[feat["week"] < before_week]
    if prior.empty:
        return CONSERVATIVE_STARTER_GAP_EPA, "TODO conservative 0.03 (no prior weeks)"
    home = prior[["game_id", "week", "home_team", "home_starter_change"]].rename(
        columns={"home_team": "team", "home_starter_change": "changed"}
    )
    away = prior[["game_id", "week", "away_team", "away_starter_change"]].rename(
        columns={"away_team": "team", "away_starter_change": "changed"}
    )
    labeled = pd.concat([home, away], ignore_index=True)
    merged = net_epa.merge(labeled, on=["game_id", "team", "week"], how="inner")
    merged = merged.loc[merged["week"] < before_week]
    if merged.empty:
        return CONSERVATIVE_STARTER_GAP_EPA, "TODO conservative 0.03 (no matched EPA)"
    starter = merged.loc[merged["changed"] == 0, "net_epa"]
    backup = merged.loc[merged["changed"] == 1, "net_epa"]
    if len(backup) < MIN_CHANGE_GAMES or starter.empty:
        return (
            CONSERVATIVE_STARTER_GAP_EPA,
            f"TODO conservative 0.03 (change games={len(backup)} < {MIN_CHANGE_GAMES})",
        )
    gap = float(starter.mean() - backup.mean())
    if gap <= 0:
        return (
            CONSERVATIVE_STARTER_GAP_EPA,
            f"TODO conservative 0.03 (data gap {gap:.4f} was non-positive)",
        )
    return gap, f"data gap={gap:.4f} net EPA/play (n_change={len(backup)}, n_stable={len(starter)})"


def apply_starter_change_layer(
    preds: pd.DataFrame,
    before_week: int,
    features: pd.DataFrame,
    net_epa: pd.DataFrame,
) -> tuple[pd.DataFrame, float, str]:
    gap, note = starter_change_gap_epa(net_epa, features, before_week)
    drop_cols = [c for c in ("home_starter_change", "away_starter_change") if c in preds.columns]
    out = preds.drop(columns=drop_cols).merge(
        features[["game_id", "home_starter_change", "away_starter_change"]],
        on="game_id",
        how="left",
    )
    for col in ("home_starter_change", "away_starter_change"):
        out[col] = out[col].fillna(0).astype(int)
    adj = gap * (
        out["away_starter_change"].astype(float) - out["home_starter_change"].astype(float)
    )
    out["predicted_diff"] = out["predicted_diff"] + adj
    return out, gap, note
