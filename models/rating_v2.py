"""v2 ratings: 2026 priors, prior/in-season blend, pass/rush matchup splits."""

import pandas as pd
from scipy.stats import linregress

from rating_v1 import (
    ROOT,
    compute_hfa,
    compute_v1_ratings,
    filter_garbage_time,
    game_level_table,
    load_pbp,
    opponent_adjust,
    season_play_ratings,
    usable_plays,
)
from validate import game_results
from validate_oos import MIN_WEEK

PRIORS_PATH = ROOT / "data" / "priors_2026.csv"
PRIOR_REGRESSION = 0.25
BLEND_FULL_GAMES = 8
PASS_TYPE = "pass"
RUSH_TYPE = "run"

V1_OOS_R = 0.4526
V1_OOS_R2 = 0.2048
V1_OOS_SLOPE = 31.0197
V1_OOS_N = 221


def shrink_to_league(values: pd.Series, regression: float = PRIOR_REGRESSION) -> pd.Series:
    """Regress toward the league mean: (1-r) * team + r * league_avg."""
    league_avg = float(values.mean())
    return (1.0 - regression) * values + regression * league_avg


def blend_ratings(prior: float, in_season: float, games_played: int, full_games: int = BLEND_FULL_GAMES) -> float:
    """Blend prior and in-season rating. Weight on current season = min(games/8, 1)."""
    weight = min(games_played / full_games, 1.0)
    return weight * in_season + (1.0 - weight) * prior


def test_blend_math() -> None:
    """Dummy check: one team, prior 0.12, three fake 2026 games."""
    prior = 0.12
    fake_games = [0.20, 0.05, 0.25]
    in_season = sum(fake_games) / len(fake_games)

    print("Blend test (dummy 2026 data, one team)")
    print(f"  prior rating:     {prior:.4f}")
    print(f"  fake game EPA:    {fake_games}")
    print(f"  in-season rating: {in_season:.4f}  (mean of {len(fake_games)} games)")
    print()
    print("  games_played  weight_on_2026  blended")
    for n in (0, 1, 3, 8, 12):
        current = in_season if n > 0 else 0.0
        blended = blend_ratings(prior, current, n)
        weight = min(n / BLEND_FULL_GAMES, 1.0)
        print(f"  {n:12d}  {weight:14.3f}  {blended:8.4f}")
    print()
    n = 3
    blended = blend_ratings(prior, in_season, n)
    weight = min(n / BLEND_FULL_GAMES, 1.0)
    expected = weight * in_season + (1.0 - weight) * prior
    print(
        f"  3-game check: {weight:.3f}*{in_season:.4f} + {1.0 - weight:.3f}*{prior:.4f} "
        f"= {expected:.4f}  (got {blended:.4f})"
    )
    if abs(blended - expected) > 1e-12:
        raise AssertionError("blend math mismatch")
    print("  blend math OK")
    print()


def build_priors_2026(filtered: pd.DataFrame) -> pd.DataFrame:
    v1 = compute_v1_ratings(filtered)
    splits = compute_split_ratings(filtered)

    priors = v1[["team", "adj_offense", "adj_defense", "adj_net"]].rename(
        columns={
            "adj_offense": "off_2025",
            "adj_defense": "def_2025",
            "adj_net": "net_2025",
        }
    )
    priors = priors.merge(splits, on="team")

    league_net = float(priors["net_2025"].mean())
    priors["league_avg_net"] = league_net
    priors["prior_net_2026"] = shrink_to_league(priors["net_2025"])

    for col in ("pass_off", "pass_def", "rush_off", "rush_def"):
        priors[f"prior_{col}_2026"] = shrink_to_league(priors[col])

    priors = priors.sort_values("prior_net_2026", ascending=False).reset_index(drop=True)
    return priors


def compute_unit_ratings(plays: pd.DataFrame, play_type: str) -> pd.DataFrame:
    unit = plays.loc[plays["play_type"] == play_type]
    games = game_level_table(unit)
    init = season_play_ratings(unit)
    adj = opponent_adjust(games, init)
    return adj[["team", "adj_offense", "adj_defense"]]


def compute_split_ratings(filtered: pd.DataFrame) -> pd.DataFrame:
    """Four SOS-adjusted ratings: pass_off, pass_def, rush_off, rush_def.

    Defense is EPA allowed per play (higher = worse).
    """
    passing = compute_unit_ratings(filtered, PASS_TYPE).rename(
        columns={"adj_offense": "pass_off", "adj_defense": "pass_def"}
    )
    rushing = compute_unit_ratings(filtered, RUSH_TYPE).rename(
        columns={"adj_offense": "rush_off", "adj_defense": "rush_def"}
    )
    return passing.merge(rushing, on="team")


def matchup_predicted_diff(
    home_pass_off: float,
    home_pass_def: float,
    home_rush_off: float,
    home_rush_def: float,
    away_pass_off: float,
    away_pass_def: float,
    away_rush_off: float,
    away_rush_def: float,
    hfa: float,
) -> float:
    """Expected home-away EPA edge from pass/rush matchups, plus HFA.

    Defense ratings are EPA allowed (higher = worse D), so an offense's
    expected EPA vs a specific defense is the average of its own off rating
    and the opponent's def rating:

        home_pass_expected = 0.5 * (home_pass_off + away_pass_def)
        home_rush_expected = 0.5 * (home_rush_off + away_rush_def)
        away_pass_expected = 0.5 * (away_pass_off + home_pass_def)
        away_rush_expected = 0.5 * (away_rush_off + home_rush_def)

        predicted_diff = (home_pass_expected + home_rush_expected)
                       - (away_pass_expected + away_rush_expected)
                       + HFA

    That is the average of pass-unit net and rush-unit net (home minus away),
    not the raw sum, so the scale stays comparable to v1 EPA/play nets.
    The user's first draft subtracted defense as if higher=better; that is
    inverted for EPA-allowed ratings, so this additive form is used instead.
    """
    home_pass_expected = 0.5 * (home_pass_off + away_pass_def)
    home_rush_expected = 0.5 * (home_rush_off + away_rush_def)
    away_pass_expected = 0.5 * (away_pass_off + home_pass_def)
    away_rush_expected = 0.5 * (away_rush_off + home_rush_def)
    return (home_pass_expected + home_rush_expected) - (
        away_pass_expected + away_rush_expected
    ) + hfa


def predict_week_splits(week_games: pd.DataFrame, splits: pd.DataFrame, hfa: float) -> pd.DataFrame:
    by_team = splits.set_index("team")
    out = week_games.copy()
    for side, prefix in (("home_team", "home"), ("away_team", "away")):
        for col in ("pass_off", "pass_def", "rush_off", "rush_def"):
            out[f"{prefix}_{col}"] = out[side].map(by_team[col])

    needed = [
        "home_pass_off",
        "home_pass_def",
        "home_rush_off",
        "home_rush_def",
        "away_pass_off",
        "away_pass_def",
        "away_rush_off",
        "away_rush_def",
    ]
    out = out.dropna(subset=needed)
    out["predicted_diff"] = (
        0.5 * (out["home_pass_off"] + out["away_pass_def"])
        + 0.5 * (out["home_rush_off"] + out["away_rush_def"])
        - 0.5 * (out["away_pass_off"] + out["home_pass_def"])
        - 0.5 * (out["away_rush_off"] + out["home_rush_def"])
        + hfa
    )
    out["hfa"] = hfa
    return out


def walk_forward_split_oos(pbp: pd.DataFrame) -> tuple[pd.DataFrame, object, pd.DataFrame]:
    plays = usable_plays(pbp).copy()
    plays["week"] = plays["week"].astype(int)
    games = game_results(pbp)
    games["week"] = games["week"].astype(int)
    last_week = int(games["week"].max())

    frames = []
    week_rows = []
    print(
        f"v2 walk-forward OOS (pass/rush splits): ratings from weeks < W, "
        f"predict week W, W = {MIN_WEEK}..{last_week}"
    )
    print()

    for week in range(MIN_WEEK, last_week + 1):
        week_games = games.loc[games["week"] == week]
        if week_games.empty:
            continue
        prior_plays = plays.loc[plays["week"] < week]
        filtered = filter_garbage_time(prior_plays)
        hfa = compute_hfa(prior_plays)
        splits = compute_split_ratings(filtered)
        preds = predict_week_splits(week_games, splits, hfa)
        frames.append(preds)
        week_rows.append(
            {
                "week": week,
                "season_type": ",".join(sorted(preds["season_type"].dropna().unique())),
                "n_games": len(preds),
                "hfa": hfa,
            }
        )
        print(f"Week {week:2d}: {len(preds):2d} games  HFA={hfa:.4f}  teams={len(splits)}")

    table = pd.concat(frames, ignore_index=True)
    fit = linregress(table["predicted_diff"], table["actual_diff"])
    return table, fit, pd.DataFrame(week_rows)


def main() -> None:
    pd.set_option("display.width", 160)
    pd.set_option("display.max_rows", 40)
    pd.set_option("display.float_format", lambda x: f"{x: .4f}")

    pbp = load_pbp()
    plays = usable_plays(pbp)
    filtered = filter_garbage_time(plays)

    # 1. 2025 finals -> 2026 priors
    priors = build_priors_2026(filtered)
    PRIORS_PATH.parent.mkdir(parents=True, exist_ok=True)
    priors.to_csv(PRIORS_PATH, index=False)
    print("2026 priors: 0.75 * 2025 rating + 0.25 * league average")
    print(f"Saved {PRIORS_PATH}")
    print(f"League-average 2025 net: {priors['league_avg_net'].iloc[0]:.4f}")
    print()
    print(
        priors[
            [
                "team",
                "net_2025",
                "prior_net_2026",
                "pass_off",
                "pass_def",
                "rush_off",
                "rush_def",
            ]
        ].to_string(index=False)
    )
    print()

    # 2. Blend function dummy test (no 2026 games yet)
    test_blend_math()

    # 4. Formula
    print("v2 matchup prediction formula (defense = EPA allowed, higher = worse):")
    print("  home_pass_expected = 0.5 * (home_pass_off + away_pass_def)")
    print("  home_rush_expected = 0.5 * (home_rush_off + away_rush_def)")
    print("  away_pass_expected = 0.5 * (away_pass_off + home_pass_def)")
    print("  away_rush_expected = 0.5 * (away_rush_off + home_rush_def)")
    print("  predicted_diff = (home_pass_expected + home_rush_expected)")
    print("                 - (away_pass_expected + away_rush_expected)")
    print("                 + HFA")
    print()

    # 5. Same walk-forward OOS as v1, but with split matchup predictions
    table, fit, week_rows = walk_forward_split_oos(pbp)
    r2 = fit.rvalue**2

    print()
    print("Games predicted per week:")
    print(week_rows.to_string(index=False))
    print()
    print(f"Pooled OOS games: {len(table)}")
    print()

    comparison = pd.DataFrame(
        {
            "metric": ["n_games", "r", "r_squared", "slope"],
            "v1_oos": [V1_OOS_N, V1_OOS_R, V1_OOS_R2, V1_OOS_SLOPE],
            "v2_oos": [len(table), fit.rvalue, r2, fit.slope],
        }
    )
    comparison["delta"] = comparison["v2_oos"] - comparison["v1_oos"]
    print("v2 matchup-split OOS vs v1 whole-team OOS")
    print(comparison.to_string(index=False))
    print()
    print(
        f"v2 vs v1 OOS: r {fit.rvalue:.4f} vs {V1_OOS_R:.4f}  |  "
        f"R^2 {r2:.4f} vs {V1_OOS_R2:.4f}  |  "
        f"slope {fit.slope:.4f} vs {V1_OOS_SLOPE:.4f}"
    )


if __name__ == "__main__":
    main()
