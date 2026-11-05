"""Unadjusted 2025 team EPA ratings (no opponent adjustment)."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PBP_PATH = ROOT / "data" / "raw" / "pbp_2025.csv"


def main() -> None:
    pbp = pd.read_csv(PBP_PATH, low_memory=False)

    plays = pbp.dropna(subset=["posteam", "defteam", "epa"])

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

    pd.set_option("display.max_rows", 40)
    pd.set_option("display.width", 80)
    pd.set_option("display.float_format", lambda x: f"{x: .4f}")

    print("2025 team EPA ratings (unadjusted)")
    print("net rating = offense EPA/play - defense EPA/play allowed")
    print()
    print(ratings.to_string(index=False))


if __name__ == "__main__":
    main()
