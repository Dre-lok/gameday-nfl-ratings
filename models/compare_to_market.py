"""Compare blended v1 ratings to this week's market odds. No bet recommendations."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "models"))
sys.path.insert(0, str(ROOT / "odds"))

from rating_v1 import compute_hfa, load_pbp, usable_plays  # noqa: E402
from rating_v2 import PRIORS_PATH, blend_ratings  # noqa: E402
from pull_odds import RAW_PATH, pull_current_week  # noqa: E402
from validate_oos import oos_calibration  # noqa: E402

CALIBRATION_PATH = ROOT / "data" / "oos_calibration.json"
PBP_2026 = ROOT / "data" / "raw" / "pbp_2026.csv"


def american_implied(odds: int | None) -> float | None:
    if odds is None or pd.isna(odds):
        return None
    odds = int(odds)
    if odds < 0:
        return abs(odds) / (abs(odds) + 100.0)
    return 100.0 / (odds + 100.0)


def no_vig_pair(p_home: float | None, p_away: float | None) -> tuple[float | None, float | None]:
    if p_home is None or p_away is None:
        return None, None
    total = p_home + p_away
    if total <= 0:
        return None, None
    return p_home / total, p_away / total


def load_calibration(force: bool = False) -> dict:
    if CALIBRATION_PATH.exists() and not force:
        return json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
    print("Computing OOS residual calibration from 2025 walk-forward (one-time)...")
    cal = oos_calibration()
    pbp = load_pbp()
    cal["hfa_epa"] = compute_hfa(usable_plays(pbp))
    CALIBRATION_PATH.write_text(json.dumps(cal, indent=2), encoding="utf-8")
    return cal


def games_played_by_team() -> dict[str, int]:
    """2026 regular-season games played. Zero until pbp_2026.csv exists."""
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


def blended_nets(priors: pd.DataFrame, played: dict[str, int]) -> dict[str, float]:
    """Prior-blended net rating. With 0 games played this is 100% 2026 prior."""
    prior = dict(zip(priors["team"], priors["prior_net_2026"]))
    nets = {}
    for team in priors["team"]:
        n = int(played.get(team, 0))
        prior_val = float(prior[team])
        if n <= 0:
            nets[team] = prior_val
        else:
            # Placeholder: 2026 in-season v1 rating is not built yet, so stay on prior.
            nets[team] = blend_ratings(prior_val, prior_val, n)
    return nets


def predicted_margin_points(predicted_diff: float, cal: dict) -> float:
    return cal["intercept"] + cal["slope"] * predicted_diff


def win_prob_from_margin(margin_points: float, sigma: float) -> float:
    """P(home wins) from expected point margin, continuity-corrected at 0.5."""
    return float(1.0 - norm.cdf(0.5, loc=margin_points, scale=sigma))


def build_comparison(odds: pd.DataFrame, nets: dict[str, float], cal: dict) -> pd.DataFrame:
    hfa = cal["hfa_epa"]
    sigma = cal["residual_std"]
    rows = []
    for game in odds.itertuples(index=False):
        home = game.home_team
        away = game.away_team
        if home not in nets or away not in nets:
            continue
        pred_diff = nets[home] - nets[away] + hfa
        pred_margin = predicted_margin_points(pred_diff, cal)
        p_home_model = win_prob_from_margin(pred_margin, sigma)
        p_away_model = 1.0 - p_home_model

        p_home_raw = american_implied(game.market_moneyline_home)
        p_away_raw = american_implied(game.market_moneyline_away)
        p_home_mkt, p_away_mkt = no_vig_pair(p_home_raw, p_away_raw)

        if p_home_mkt is None:
            # Fallback: spread as expected margin, same residual sigma, no ML vig to remove
            p_home_mkt = win_prob_from_margin(-float(game.market_spread), sigma)
            p_away_mkt = 1.0 - p_home_mkt

        home_spread = float(game.market_spread)
        pred_home_spread = -pred_margin

        rows.append(
            {
                "game": game.game,
                "team": home,
                "side": "home",
                "your_prob": p_home_model,
                "market_prob": p_home_mkt,
                "edge": p_home_model - p_home_mkt,
                "market_spread": home_spread,
                "your_predicted_spread": pred_home_spread,
            }
        )
        rows.append(
            {
                "game": game.game,
                "team": away,
                "side": "away",
                "your_prob": p_away_model,
                "market_prob": p_away_mkt,
                "edge": p_away_model - p_away_mkt,
                "market_spread": -home_spread,
                "your_predicted_spread": -pred_home_spread,
            }
        )
    table = pd.DataFrame(rows)
    table["abs_edge"] = table["edge"].abs()
    table = table.sort_values("abs_edge", ascending=False).reset_index(drop=True)
    return table


def main() -> None:
    if RAW_PATH.exists():
        payload = json.loads(RAW_PATH.read_text(encoding="utf-8"))
        from pull_odds import parse_games

        odds = parse_games(payload.get("scoreboard") or payload)
        slate = payload.get("slate", "saved current_week.json")
        print(f"Loaded odds from {RAW_PATH}")
    else:
        odds, slate = pull_current_week()
        print(f"Pulled odds. Saved {RAW_PATH}")

    priors = pd.read_csv(PRIORS_PATH)
    played = games_played_by_team()
    n_played = sum(played.values())
    nets = blended_nets(priors, played)
    cal = load_calibration()

    table = build_comparison(odds, nets, cal)

    pd.set_option("display.width", 160)
    pd.set_option("display.max_rows", 80)
    pd.set_option("display.float_format", lambda x: f"{x: .4f}")

    print(f"Slate: {slate}")
    print(
        f"Model: v1 net + 2026 prior blend. 2026 games in pbp: {n_played}. "
        f"games_played=0 => 100% prior."
    )
    print(
        f"Calibration (2025 OOS): slope={cal['slope']:.2f} pts/rating, "
        f"intercept={cal['intercept']:.2f}, residual_std={cal['residual_std']:.2f}, "
        f"HFA={cal['hfa_epa']:.4f} EPA"
    )
    print("Market win prob: no-vig moneyline (raw implied / sum of both sides).")
    print("Your win prob: Normal(predicted_margin, residual_std), P(margin > 0.5).")
    print("Spread is that team's line (negative = team favored). Comparison only — no bets.")
    print()

    show = table[
        [
            "team",
            "game",
            "your_prob",
            "market_prob",
            "edge",
            "market_spread",
            "your_predicted_spread",
        ]
    ].copy()
    show["your_prob"] = show["your_prob"].map(lambda x: f"{100 * x:5.1f}%")
    show["market_prob"] = show["market_prob"].map(lambda x: f"{100 * x:5.1f}%")
    show["edge"] = show["edge"].map(lambda x: f"{100 * x:+5.1f} pp")
    show["market_spread"] = show["market_spread"].map(lambda x: f"{x:+.1f}")
    show["your_predicted_spread"] = show["your_predicted_spread"].map(lambda x: f"{x:+.1f}")
    print(show.to_string(index=False))
    print()
    print(f"{len(table)} team-rows from {len(odds)} games, sorted by |edge| descending.")


if __name__ == "__main__":
    main()
