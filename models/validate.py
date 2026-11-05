"""In-sample check: do v1 ratings plus league HFA predict actual score margins?"""

import pandas as pd
from scipy.stats import linregress

from rating_v1 import (
    compute_hfa,
    compute_v1_ratings,
    filter_garbage_time,
    load_pbp,
    usable_plays,
)


def game_results(pbp: pd.DataFrame) -> pd.DataFrame:
    games = (
        pbp.dropna(subset=["game_id", "home_team", "away_team", "home_score", "away_score"])
        .drop_duplicates("game_id")
        [
            [
                "game_id",
                "week",
                "season_type",
                "home_team",
                "away_team",
                "home_score",
                "away_score",
            ]
        ]
        .copy()
    )
    games["actual_diff"] = games["home_score"] - games["away_score"]
    return games


def build_prediction_table(
    pbp: pd.DataFrame, ratings: pd.DataFrame, hfa: float
) -> pd.DataFrame:
    net = dict(zip(ratings["team"], ratings["adj_net"]))
    games = game_results(pbp)
    games["home_net"] = games["home_team"].map(net)
    games["away_net"] = games["away_team"].map(net)
    games = games.dropna(subset=["home_net", "away_net"])
    games["predicted_diff"] = games["home_net"] - games["away_net"] + hfa
    return games.reset_index(drop=True)


def main() -> None:
    pbp = load_pbp()
    plays = usable_plays(pbp)
    filtered = filter_garbage_time(plays)
    hfa = compute_hfa(plays)
    ratings = compute_v1_ratings(filtered)
    table = build_prediction_table(pbp, ratings, hfa)

    fit = linregress(table["predicted_diff"], table["actual_diff"])
    r_squared = fit.rvalue**2

    preview = table[["game_id", "home_team", "away_team", "predicted_diff", "actual_diff"]]
    pd.set_option("display.width", 140)
    pd.set_option("display.float_format", lambda x: f"{x: .4f}")

    print("League-wide home-field advantage (home EPA/play - away EPA/play)")
    print(f"{hfa:.4f}")
    print("(Computed on all non-null EPA plays, not garbage-time filtered.)")
    print()
    print(f"Games in validation table: {len(table)}")
    print("One row per game: predicted_diff = (home_net - away_net) + HFA")
    print("actual_diff = home_score - away_score")
    print()
    print("First 8 games:")
    print(preview.head(8).to_string(index=False))
    print()
    print("Last 8 games:")
    print(preview.tail(8).to_string(index=False))
    print()
    print("Linear regression: actual_diff ~ predicted_diff")
    print(f"correlation r: {fit.rvalue:.4f}")
    print(f"slope (actual points per rating point): {fit.slope:.4f}")
    print(f"r-squared: {r_squared:.4f}")


if __name__ == "__main__":
    main()
