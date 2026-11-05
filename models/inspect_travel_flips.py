"""Break down v3_travel_refined SU gain: pick flips and margin MAE. Do not ship."""

from __future__ import annotations

import json

import pandas as pd

from calibration_check import predicted_winner_table
from rating_v1 import ROOT, load_pbp
from test_levers import cal_from_predictions, pick_rates
from validate_oos import run_walk_forward

CAL_PATH = ROOT / "data" / "oos_calibration.json"


def main() -> None:
    pbp = load_pbp()
    stored = json.loads(CAL_PATH.read_text(encoding="utf-8"))
    print("v3_travel_refined investigation - same isolated walk-forward as the original test")
    print("QB-change production flag is NOT applied here (do not bundle).")
    print()

    off_table, off_fit, _ = run_walk_forward(
        quiet=True,
        use_recency_weight=True,
        use_turnover_adjustment=True,
        use_rest_travel=False,
        lever=None,
        use_qb_change=False,
        pbp=pbp,
    )
    on_table, on_fit, _ = run_walk_forward(
        quiet=True,
        use_recency_weight=True,
        use_turnover_adjustment=True,
        use_rest_travel=False,
        lever="travel_refined",
        use_qb_change=False,
        pbp=pbp,
    )

    cal_off = cal_from_predictions(off_table)
    cal_on = cal_from_predictions(on_table)
    off_scored = predicted_winner_table(off_table, cal_off)
    on_scored = predicted_winner_table(on_table, cal_on)
    merged = off_scored.merge(
        on_scored,
        on="game_id",
        suffixes=("_off", "_on"),
    )
    n = len(merged)
    su_off, ats_off, _, _ = pick_rates(off_table)
    su_on, ats_on, _, _ = pick_rates(on_table)

    flipped = merged["predicted_home_off"] != merged["predicted_home_on"]
    flips = merged.loc[flipped].copy()
    n_flips = int(len(flips))
    n_correct = int(flips["predicted_winner_won_on"].sum()) if n_flips else 0
    n_incorrect = n_flips - n_correct
    off_right_on_wrong = (
        int((flips["predicted_winner_won_off"] & ~flips["predicted_winner_won_on"]).sum())
        if n_flips
        else 0
    )
    off_wrong_on_right = (
        int((~flips["predicted_winner_won_off"] & flips["predicted_winner_won_on"]).sum())
        if n_flips
        else 0
    )

    mae_off_refit = float((off_scored["pred_margin"] - off_scored["actual_diff"]).abs().mean())
    mae_on_refit = float((on_scored["pred_margin"] - on_scored["actual_diff"]).abs().mean())

    def stored_margin(pred_diff: pd.Series) -> pd.Series:
        return stored["intercept"] + stored["slope"] * pred_diff

    mae_off_stored = float(
        (stored_margin(off_table["predicted_diff"]) - off_table["actual_diff"]).abs().mean()
    )
    mae_on_stored = float(
        (stored_margin(on_table["predicted_diff"]) - on_table["actual_diff"]).abs().mean()
    )

    print("ON vs OFF (travel_refined, isolated)")
    print(
        f"OFF  r={off_fit.rvalue:.4f}  R^2={off_fit.rvalue**2:.4f}  "
        f"SU={100 * su_off:.1f}% ({n})  ATS={100 * ats_off:.1f}%"
    )
    print(
        f"ON   r={on_fit.rvalue:.4f}  R^2={on_fit.rvalue**2:.4f}  "
        f"SU={100 * su_on:.1f}% ({n})  ATS={100 * ats_on:.1f}%"
    )
    print()
    print("Straight-up pick flips (who the model favors, p_home >= 0.5)")
    print(f"  games: {n}")
    print(f"  flips: {n_flips}")
    print(f"  flips where ON pick was correct: {n_correct}")
    print(f"  flips where ON pick was incorrect: {n_incorrect}")
    print(f"  of those, OFF wrong -> ON right: {off_wrong_on_right}")
    print(f"  of those, OFF right -> ON wrong: {off_right_on_wrong}")
    print(
        f"  net SU games from flips: {off_wrong_on_right - off_right_on_wrong}  "
        f"(SU delta {100 * (su_on - su_off):+.1f}pp ~ {(su_on - su_off) * n:+.1f} games)"
    )
    print()
    print("Mean |predicted margin - actual margin| (points)")
    print(f"  MAE using each run's own OOS fit:  OFF {mae_off_refit:.3f}  ON {mae_on_refit:.3f}  "
          f"delta {mae_on_refit - mae_off_refit:+.3f}")
    print(f"  MAE using stored dashboard cal:    OFF {mae_off_stored:.3f}  ON {mae_on_stored:.3f}  "
          f"delta {mae_on_stored - mae_off_stored:+.3f}")
    if n_flips:
        print()
        print("Flip list (week, matchup, OFF pick, ON pick, actual winner, ON correct)")
        flips = flips.copy()
        flips["actual_winner"] = "HOME"
        flips.loc[flips["actual_diff_on"] < 0, "actual_winner"] = "AWAY"
        flips.loc[flips["actual_diff_on"] == 0, "actual_winner"] = "TIE"
        flips["off_pick"] = flips["predicted_home_off"].map({True: "HOME", False: "AWAY"})
        flips["on_pick"] = flips["predicted_home_on"].map({True: "HOME", False: "AWAY"})
        show_cols = [
            "week_on",
            "away_team_on",
            "home_team_on",
            "off_pick",
            "on_pick",
            "actual_winner",
            "predicted_winner_won_on",
        ]
        week_col = "week_on" if "week_on" in flips.columns else "week_off"
        if week_col != "week_on" and "week_on" not in show_cols:
            pass
        keep = [c for c in show_cols if c in flips.columns]
        pd.set_option("display.width", 140)
        pd.set_option("display.max_rows", 200)
        print(flips[keep].to_string(index=False))


if __name__ == "__main__":
    main()
