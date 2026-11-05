"""Walk-forward out-of-sample validation of v1 ratings."""

from __future__ import annotations

import argparse

import pandas as pd
from scipy.stats import linregress

from metrics_log import log_run
from rating_v1 import (
    ROOT,
    compute_hfa,
    compute_v1_ratings,
    filter_garbage_time,
    load_pbp,
    usable_plays,
)
from rest_travel import (  # noqa: E402
    adjustment_from_prior,
    merge_features,
    rest_travel_features,
)
from levers import (  # noqa: E402
    apply_qb_change_layer,
    ols_epa_adjustment,
    qb_change_features,
    rest_continuous_features,
    team_game_off_epa,
    travel_refined_features,
)
from model_config import USE_QB_CHANGE as CONFIG_USE_QB_CHANGE  # noqa: E402
from starter_change import (  # noqa: E402
    apply_starter_change_layer,
    starter_change_features,
    team_game_net_epa,
)
from weather import weather_features  # noqa: E402
from validate import game_results

MIN_WEEK = 5

IN_SAMPLE_R = 0.5883
IN_SAMPLE_R2 = 0.3461
IN_SAMPLE_SLOPE = 42.3156
IN_SAMPLE_N = 285

OOS_PREDICTIONS_PATH = ROOT / "data" / "oos_predictions.csv"


def layers_active_label(
    use_recency_weight: bool,
    use_turnover_adjustment: bool,
    use_rest_travel: bool = False,
    lever: str | None = None,
    use_qb_change: bool = False,
) -> str:
    active = []
    if use_recency_weight:
        active.append("recency")
    if use_turnover_adjustment:
        active.append("turnover_adjustment")
    if use_qb_change or lever == "qb_change":
        active.append("qb_change")
    if use_rest_travel:
        active.append("rest_travel")
    if lever and lever != "qb_change":
        active.append(lever)
    return ",".join(active) if active else "none"


def default_version_tag(
    use_recency_weight: bool,
    use_turnover_adjustment: bool,
    use_rest_travel: bool = False,
    use_qb_change: bool = False,
) -> str:
    if use_rest_travel and use_recency_weight and use_turnover_adjustment:
        return "v3_rest_travel"
    if use_qb_change and use_recency_weight and use_turnover_adjustment:
        return "v3_qb_change"
    if use_recency_weight and use_turnover_adjustment:
        return "v2_default_recency_turnover"
    if use_recency_weight:
        return "v1_recency"
    if use_turnover_adjustment:
        return "v1_turnover_adj"
    return "v1_flags_off"


def ratings_before_week(
    plays: pd.DataFrame,
    week: int,
    use_recency_weight: bool = True,
    use_turnover_adjustment: bool = True,
) -> tuple[pd.DataFrame, float]:
    """v1 ratings + HFA using only plays from weeks strictly before `week`.

    Rest/travel is a prediction-time layer and does not change these ratings.
    """
    prior = plays.loc[plays["week"] < week]
    filtered = filter_garbage_time(prior)
    hfa = compute_hfa(prior)
    ratings = compute_v1_ratings(
        filtered,
        use_recency_weight=use_recency_weight,
        use_turnover_adjustment=use_turnover_adjustment,
    )
    return ratings, hfa


def _resolve_qb_change(
    use_qb_change: bool | None, lever: str | None, use_rest_travel: bool
) -> bool:
    isolated_other = use_rest_travel or lever in ("rest_continuous", "travel_refined")
    if isolated_other:
        if use_qb_change:
            raise ValueError("Do not bundle levers; test one flag at a time.")
        return False
    if lever == "qb_change":
        return True
    if use_qb_change is None:
        return bool(CONFIG_USE_QB_CHANGE)
    return bool(use_qb_change)


def run_walk_forward(
    quiet: bool = False,
    use_recency_weight: bool = True,
    use_turnover_adjustment: bool = True,
    use_rest_travel: bool = False,
    lever: str | None = None,
    use_qb_change: bool | None = None,
    pbp: pd.DataFrame | None = None,
):
    """Ratings from weeks < W, predict week W, for W = 5 through season end.

    `lever` is one isolated extra flag. starter_change and weather are tested
    on top of production qb_change. rest/travel stay mutually exclusive with
    qb_change so old isolation tests remain valid.
    """
    if lever and use_rest_travel:
        raise ValueError("Do not bundle levers; test one flag at a time.")
    apply_qb = _resolve_qb_change(use_qb_change, lever, use_rest_travel)
    if pbp is None:
        pbp = load_pbp()
    plays = usable_plays(pbp).copy()
    plays["week"] = plays["week"].astype(int)
    games = game_results(pbp)
    games["week"] = games["week"].astype(int)

    features = None
    qb_features = None
    off_epa = None
    starter_features = None
    net_epa = None
    if apply_qb:
        qb_features = qb_change_features(pbp)
        off_epa = team_game_off_epa(pbp)
    if lever == "starter_change":
        starter_features = starter_change_features(pbp)
        net_epa = team_game_net_epa(pbp)
    if lever == "weather":
        features = weather_features(pbp)
    if lever == "rest_continuous":
        features = rest_continuous_features(pbp)
    elif lever == "travel_refined":
        features = travel_refined_features(pbp)
    elif use_rest_travel:
        features = rest_travel_features(pbp)
    elif lever not in (None, "qb_change", "starter_change", "weather"):
        raise ValueError(f"Unknown lever: {lever}")

    last_week = int(games["week"].max())
    frames = []
    week_rows = []
    if not quiet:
        layers = layers_active_label(
            use_recency_weight,
            use_turnover_adjustment,
            use_rest_travel,
            lever,
            use_qb_change=apply_qb,
        )
        print(
            f"Walk-forward OOS: ratings from weeks < W, predict week W, "
            f"W = {MIN_WEEK}..{last_week}  layers={layers}"
        )
        print()

    for week in range(MIN_WEEK, last_week + 1):
        week_games = games.loc[games["week"] == week]
        if week_games.empty:
            continue
        ratings, hfa = ratings_before_week(
            plays,
            week,
            use_recency_weight=use_recency_weight,
            use_turnover_adjustment=use_turnover_adjustment,
        )
        preds = predict_week(week_games, ratings, hfa)
        if apply_qb and qb_features is not None and off_epa is not None:
            preds, _, _ = apply_qb_change_layer(preds, week, qb_features, off_epa)
        if lever == "starter_change" and starter_features is not None and net_epa is not None:
            preds, _, _ = apply_starter_change_layer(
                preds, week, starter_features, net_epa
            )
        elif lever == "weather" and features is not None:
            prior_games = games.loc[games["week"] < week]
            prior_preds = predict_week(prior_games, ratings, hfa)
            if apply_qb and qb_features is not None and off_epa is not None:
                prior_preds, _, _ = apply_qb_change_layer(
                    prior_preds, week, qb_features, off_epa
                )
            prior_preds = prior_preds.merge(features, on="game_id", how="left")
            preds = preds.merge(features, on="game_id", how="left")
            prior_preds["weather_adverse"] = prior_preds["weather_adverse"].fillna(0).astype(int)
            preds["weather_adverse"] = preds["weather_adverse"].fillna(0).astype(int)
            adj = ols_epa_adjustment(prior_preds, preds, ["weather_adverse"])
            preds["predicted_diff"] = preds["predicted_diff"] + adj
        elif lever == "rest_continuous" and features is not None:
            prior_games = games.loc[games["week"] < week]
            prior_preds = predict_week(prior_games, ratings, hfa)
            prior_preds = prior_preds.merge(features, on="game_id", how="left")
            preds = preds.merge(features, on="game_id", how="left")
            for col in ("home_rest_vs_std", "away_rest_vs_std"):
                prior_preds[col] = prior_preds[col].fillna(0.0)
                preds[col] = preds[col].fillna(0.0)
            adj = ols_epa_adjustment(
                prior_preds, preds, ["home_rest_vs_std", "away_rest_vs_std"]
            )
            preds["predicted_diff"] = preds["predicted_diff"] + adj
        elif lever == "travel_refined" and features is not None:
            prior_games = games.loc[games["week"] < week]
            prior_preds = predict_week(prior_games, ratings, hfa)
            prior_preds = prior_preds.merge(features, on="game_id", how="left")
            preds = preds.merge(features, on="game_id", how="left")
            cols = ["away_travel_kmi", "away_tz_shift_3", "away_denver_altitude"]
            for col in cols:
                prior_preds[col] = prior_preds[col].fillna(0.0)
                preds[col] = preds[col].fillna(0.0)
            adj = ols_epa_adjustment(prior_preds, preds, cols)
            preds["predicted_diff"] = preds["predicted_diff"] + adj
        elif use_rest_travel and features is not None:
            prior_games = games.loc[games["week"] < week]
            prior_preds = predict_week(prior_games, ratings, hfa)
            prior_preds = merge_features(prior_preds, features)
            preds = merge_features(preds, features)
            adj = adjustment_from_prior(prior_preds, preds)
            preds["predicted_diff"] = preds["predicted_diff"] + adj
        frames.append(preds)
        week_rows.append(
            {
                "week": week,
                "season_type": ",".join(sorted(preds["season_type"].dropna().unique())),
                "n_games": len(preds),
                "hfa": hfa,
            }
        )
        if not quiet:
            print(f"Week {week:2d}: {len(preds):2d} games  HFA={hfa:.4f}  teams={len(ratings)}")

    table = pd.concat(frames, ignore_index=True)
    fit = linregress(table["predicted_diff"], table["actual_diff"])
    return table, fit, pd.DataFrame(week_rows)


def oos_calibration() -> dict:
    """Slope/intercept/residual std from the v1 walk-forward (actual points vs predicted_diff)."""
    table, fit, _ = run_walk_forward(quiet=True)
    fitted = fit.intercept + fit.slope * table["predicted_diff"]
    resid = table["actual_diff"] - fitted
    return {
        "n": int(len(table)),
        "slope": float(fit.slope),
        "intercept": float(fit.intercept),
        "r": float(fit.rvalue),
        "r_squared": float(fit.rvalue**2),
        "residual_std": float(resid.std(ddof=1)),
    }


def predict_week(week_games: pd.DataFrame, ratings: pd.DataFrame, hfa: float) -> pd.DataFrame:
    net = dict(zip(ratings["team"], ratings["adj_net"]))
    out = week_games.copy()
    out["home_net"] = out["home_team"].map(net)
    out["away_net"] = out["away_team"].map(net)
    out = out.dropna(subset=["home_net", "away_net"])
    out["predicted_diff"] = out["home_net"] - out["away_net"] + hfa
    out["hfa"] = hfa
    return out


def _print_summary(table: pd.DataFrame, fit, week_rows: pd.DataFrame) -> None:
    r2 = fit.rvalue**2
    pd.set_option("display.width", 120)
    pd.set_option("display.float_format", lambda x: f"{x: .4f}")

    print()
    print("Games predicted per week:")
    print(week_rows.to_string(index=False))
    print()
    print(f"Pooled OOS games: {len(table)}")
    print("Linear regression: actual_diff ~ predicted_diff")
    print()

    comparison = pd.DataFrame(
        {
            "metric": ["n_games", "r", "r_squared", "slope"],
            "in_sample": [IN_SAMPLE_N, IN_SAMPLE_R, IN_SAMPLE_R2, IN_SAMPLE_SLOPE],
            "out_of_sample": [len(table), fit.rvalue, r2, fit.slope],
        }
    )
    comparison["drop"] = comparison["out_of_sample"] - comparison["in_sample"]
    print(comparison.to_string(index=False))
    print()
    print(
        f"OOS vs in-sample: r {fit.rvalue:.4f} vs {IN_SAMPLE_R:.4f}  |  "
        f"R^2 {r2:.4f} vs {IN_SAMPLE_R2:.4f}  |  "
        f"slope {fit.slope:.4f} vs {IN_SAMPLE_SLOPE:.4f}"
    )


def run_and_log(
    use_recency_weight: bool = True,
    use_turnover_adjustment: bool = True,
    use_rest_travel: bool = False,
    version_tag: str | None = None,
    notes: str = "",
    quiet: bool = False,
    pbp: pd.DataFrame | None = None,
    save_predictions: bool = True,
    pick_accuracy_su=None,
    pick_accuracy_ats=None,
):
    apply_qb = _resolve_qb_change(None, None, use_rest_travel)
    table, fit, week_rows = run_walk_forward(
        quiet=quiet,
        use_recency_weight=use_recency_weight,
        use_turnover_adjustment=use_turnover_adjustment,
        use_rest_travel=use_rest_travel,
        use_qb_change=apply_qb,
        pbp=pbp,
    )
    if not quiet:
        _print_summary(table, fit, week_rows)

    tag = version_tag or default_version_tag(
        use_recency_weight,
        use_turnover_adjustment,
        use_rest_travel,
        use_qb_change=apply_qb,
    )
    layers = layers_active_label(
        use_recency_weight,
        use_turnover_adjustment,
        use_rest_travel,
        use_qb_change=apply_qb,
    )
    log_run(
        version_tag=tag,
        layers_active=layers,
        games=len(table),
        r=float(fit.rvalue),
        r_squared=float(fit.rvalue**2),
        slope=float(fit.slope),
        notes=notes,
        pick_accuracy_su=pick_accuracy_su,
        pick_accuracy_ats=pick_accuracy_ats,
    )
    if save_predictions:
        OOS_PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(OOS_PREDICTIONS_PATH, index=False)
    return table, fit, week_rows


def main(
    use_recency_weight: bool = True,
    use_turnover_adjustment: bool = True,
    use_rest_travel: bool = False,
    version_tag: str | None = None,
    notes: str = "",
    quiet: bool = False,
) -> None:
    run_and_log(
        use_recency_weight=use_recency_weight,
        use_turnover_adjustment=use_turnover_adjustment,
        use_rest_travel=use_rest_travel,
        version_tag=version_tag,
        notes=notes,
        quiet=quiet,
        save_predictions=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Walk-forward OOS validation of v1 ratings.")
    parser.add_argument(
        "--no-recency-weight",
        action="store_true",
        help="Disable recency weighting (on by default).",
    )
    parser.add_argument(
        "--no-turnover-adjustment",
        action="store_true",
        help="Disable turnover EPA regression (on by default).",
    )
    parser.add_argument(
        "--rest-travel",
        action="store_true",
        help="Enable rest/travel prediction adjustment (off by default).",
    )
    parser.add_argument("--version-tag", default=None, help="Override metrics_log version_tag.")
    parser.add_argument("--notes", default="", help="Optional notes stored in metrics_log.")
    parser.add_argument("--quiet", action="store_true", help="Skip per-week printout.")
    args = parser.parse_args()
    main(
        use_recency_weight=not args.no_recency_weight,
        use_turnover_adjustment=not args.no_turnover_adjustment,
        use_rest_travel=args.rest_travel,
        version_tag=args.version_tag,
        notes=args.notes,
        quiet=args.quiet,
    )
