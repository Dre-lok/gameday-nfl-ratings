"""Filtered + opponent-adjusted 2025 team EPA ratings."""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PBP_PATH = ROOT / "data" / "raw" / "pbp_2025.csv"
N_ITERATIONS = 8
WP_LOW = 0.05
WP_HIGH = 0.95
RECENCY_HALF_LIFE_GAMES = 6.0
TURNOVER_REGRESSION = 0.5


def season_play_ratings(plays: pd.DataFrame) -> pd.DataFrame:
    """Play-weighted offense / defense / net EPA, same grouping as rating_v0."""
    offense = (
        plays.groupby("posteam", as_index=False)["epa"]
        .mean()
        .rename(columns={"posteam": "team", "epa": "offense_epa"})
    )
    defense = (
        plays.groupby("defteam", as_index=False)["epa"]
        .mean()
        .rename(columns={"defteam": "team", "epa": "defense_epa"})
    )
    ratings = offense.merge(defense, on="team")
    ratings["net_rating"] = ratings["offense_epa"] - ratings["defense_epa"]
    ratings = ratings.sort_values("net_rating", ascending=False).reset_index(drop=True)
    ratings.insert(0, "rank", range(1, len(ratings) + 1))
    return ratings


def game_level_table(plays: pd.DataFrame) -> pd.DataFrame:
    """One row per team per game: opponent and that game's offensive EPA."""
    agg = {"epa": "mean"}
    if "week" in plays.columns:
        agg["week"] = "first"
    game_off = (
        plays.groupby(["game_id", "posteam", "defteam"], as_index=False)
        .agg(agg)
        .rename(columns={"posteam": "team", "defteam": "opponent", "epa": "off_epa"})
    )
    opp_off = game_off[["game_id", "team", "off_epa"]].rename(
        columns={"team": "opponent", "off_epa": "epa_allowed"}
    )
    games = game_off.merge(opp_off, on=["game_id", "opponent"], how="inner")
    return games


def load_pbp() -> pd.DataFrame:
    return pd.read_csv(PBP_PATH, low_memory=False)


def usable_plays(pbp: pd.DataFrame) -> pd.DataFrame:
    return pbp.dropna(subset=["posteam", "defteam", "epa"])


def filter_garbage_time(plays: pd.DataFrame) -> pd.DataFrame:
    return plays[plays["wp"].between(WP_LOW, WP_HIGH, inclusive="both")].copy()


def compute_hfa(plays: pd.DataFrame) -> float:
    """League-wide home EPA/play minus away EPA/play. Single constant, not per-team."""
    home_epa = plays.loc[plays["posteam_type"] == "home", "epa"].mean()
    away_epa = plays.loc[plays["posteam_type"] == "away", "epa"].mean()
    return float(home_epa - away_epa)


def _turnover_mask(plays: pd.DataFrame) -> pd.Series:
    interception = pd.to_numeric(plays.get("interception"), errors="coerce").fillna(0)
    fumble_lost = pd.to_numeric(plays.get("fumble_lost"), errors="coerce").fillna(0)
    return (interception == 1) | (fumble_lost == 1)


def apply_turnover_adjustment(
    plays: pd.DataFrame, regression: float = TURNOVER_REGRESSION
) -> pd.DataFrame:
    """Regress turnover-related EPA 50% toward league-average EPA.

    On interception / fumble-lost plays the gap vs league-average EPA is
    treated as the turnover portion; that portion is pulled halfway to zero
    (i.e. play EPA is pulled halfway to the league mean). Other plays unchanged.
    """
    out = plays.copy()
    is_to = _turnover_mask(out)
    if not bool(is_to.any()):
        return out
    league_avg = float(out["epa"].mean())
    out.loc[is_to, "epa"] = (1.0 - regression) * out.loc[is_to, "epa"] + regression * league_avg
    return out


def add_recency_weights(
    games: pd.DataFrame, half_life: float = RECENCY_HALF_LIFE_GAMES
) -> pd.DataFrame:
    """Exponential recency weights; most recent game for a team has weight 1.

    Half-life is in games (default 6), not weeks, so a bye does not extra-decay.
    """
    out = games.copy()
    if "week" not in out.columns:
        out["week"] = out["game_id"].astype(str).str.split("_").str[1].astype(int)
    out["recency_weight"] = 1.0
    for _, group in out.groupby("team"):
        order = group.sort_values(["week", "game_id"]).index
        n = len(order)
        games_ago = np.arange(n - 1, -1, -1, dtype=float)
        out.loc[order, "recency_weight"] = 0.5 ** (games_ago / half_life)
    return out


def compute_v1_ratings(
    filtered: pd.DataFrame,
    use_recency_weight: bool = True,
    use_turnover_adjustment: bool = True,
) -> pd.DataFrame:
    plays = apply_turnover_adjustment(filtered) if use_turnover_adjustment else filtered
    games = game_level_table(plays)
    if use_recency_weight:
        games = add_recency_weights(games)
    init_ratings = season_play_ratings(plays)
    return opponent_adjust(games, init_ratings)


def opponent_adjust(games: pd.DataFrame, init_ratings: pd.DataFrame) -> pd.DataFrame:
    """SRS-style simultaneous opponent adjustment, 8 iterations."""
    offense_rating = dict(zip(init_ratings["team"], init_ratings["offense_epa"]))
    defense_rating = dict(zip(init_ratings["team"], init_ratings["defense_epa"]))
    teams = list(init_ratings["team"])
    weighted = "recency_weight" in games.columns

    for _ in range(N_ITERATIONS):
        league_avg_off = sum(offense_rating.values()) / len(offense_rating)
        league_avg_def = sum(defense_rating.values()) / len(defense_rating)

        new_off = {}
        new_def = {}
        for team in teams:
            team_games = games.loc[games["team"] == team]
            opp_def = team_games["opponent"].map(defense_rating)
            opp_off = team_games["opponent"].map(offense_rating)
            off_vals = team_games["off_epa"] - opp_def
            def_vals = team_games["epa_allowed"] - opp_off
            if weighted:
                weights = team_games["recency_weight"].to_numpy(dtype=float)
                new_off[team] = float(np.average(off_vals, weights=weights)) + league_avg_def
                new_def[team] = float(np.average(def_vals, weights=weights)) + league_avg_off
            else:
                new_off[team] = off_vals.mean() + league_avg_def
                new_def[team] = def_vals.mean() + league_avg_off

        offense_rating = new_off
        defense_rating = new_def

    adjusted = pd.DataFrame(
        {
            "team": teams,
            "adj_offense": [offense_rating[t] for t in teams],
            "adj_defense": [defense_rating[t] for t in teams],
        }
    )
    adjusted["adj_net"] = adjusted["adj_offense"] - adjusted["adj_defense"]
    adjusted = adjusted.sort_values("adj_net", ascending=False).reset_index(drop=True)
    adjusted.insert(0, "rank", range(1, len(adjusted) + 1))
    return adjusted


def main() -> None:
    pbp = load_pbp()
    plays = usable_plays(pbp)
    v0_ratings = season_play_ratings(plays)

    before_wp = len(plays)
    filtered = filter_garbage_time(plays)
    dropped = before_wp - len(filtered)
    hfa = compute_hfa(plays)

    print("Garbage-time filter: drop plays with wp < 0.05 or wp > 0.95")
    print(f"Rows before WP filter (non-null posteam/defteam/epa): {before_wp}")
    print(f"Rows dropped: {dropped}")
    print(f"Rows remaining: {len(filtered)}")
    print()
    print(f"League-wide home-field advantage (home EPA/play - away EPA/play): {hfa:.4f}")
    print("(Computed on all non-null EPA plays, not garbage-time filtered.)")
    print()

    v1_ratings = compute_v1_ratings(filtered)

    comparison = v1_ratings.merge(
        v0_ratings[["rank", "team", "offense_epa", "defense_epa", "net_rating"]].rename(
            columns={
                "rank": "v0_rank",
                "offense_epa": "v0_offense",
                "defense_epa": "v0_defense",
                "net_rating": "v0_net",
            }
        ),
        on="team",
    )
    comparison["rank_change"] = comparison["v0_rank"] - comparison["rank"]

    pd.set_option("display.max_rows", 40)
    pd.set_option("display.width", 140)
    pd.set_option("display.float_format", lambda x: f"{x: .4f}")

    print("rating_v0 — unadjusted, all non-null EPA plays")
    print("net = offense EPA/play - defense EPA/play allowed")
    print()
    print(v0_ratings.to_string(index=False))
    print()
    print("rating_v1 — garbage-time filtered + 8-iteration opponent adjustment")
    print("Sorted 1-32 by adjusted net rating. rank_change: + means moved up vs v0.")
    print()
    print(
        comparison[
            [
                "rank",
                "team",
                "adj_offense",
                "adj_defense",
                "adj_net",
                "v0_rank",
                "v0_net",
                "rank_change",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
