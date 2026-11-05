"""Filtered OOS backtest: straight-up winner accuracy and against-the-spread."""

from __future__ import annotations

import pandas as pd
from scipy.stats import linregress

from calibration_check import bucket_label, load_oos_predictions, predicted_winner_table
from metrics_log import LOG_PATH, log_run
from rating_v1 import load_pbp

EXCLUDE_BUCKET = "50-60%"
BREAKEVEN = 0.524


def cal_from_predictions(preds) -> dict:
    """Slope / intercept / residual std from this OOS table (same as oos_calibration)."""
    fit = linregress(preds["predicted_diff"], preds["actual_diff"])
    fitted = fit.intercept + fit.slope * preds["predicted_diff"]
    resid = preds["actual_diff"] - fitted
    return {
        "n": int(len(preds)),
        "slope": float(fit.slope),
        "intercept": float(fit.intercept),
        "r": float(fit.rvalue),
        "r_squared": float(fit.rvalue**2),
        "residual_std": float(resid.std(ddof=1)),
    }


def load_closing_spreads() -> pd.DataFrame:
    """nflverse spread_line: positive means the home team is favored by that many points."""
    pbp = load_pbp()
    return (
        pbp.dropna(subset=["game_id", "spread_line"])
        .drop_duplicates("game_id")[["game_id", "spread_line"]]
        .copy()
    )


def attach_ats(scored: pd.DataFrame, spreads: pd.DataFrame) -> pd.DataFrame:
    """Grade whether the model's predicted margin would have covered the close.

    Model predicted margin is already in points (calibrated). Home covers if
    actual_diff > spread_line; the model picks home to cover if pred_margin
    > spread_line. Pushes (actual = number) and pick'ems (model = number)
    are left ungraded.
    """
    out = scored.merge(spreads, on="game_id", how="left")
    out["model_picks_home_cover"] = out["pred_margin"] > out["spread_line"]
    out["home_covered"] = out["actual_diff"] > out["spread_line"]
    out["ats_push"] = out["actual_diff"] == out["spread_line"]
    out["ats_no_pick"] = out["pred_margin"] == out["spread_line"]
    out["ats_gradable"] = (
        out["spread_line"].notna() & ~out["ats_push"] & ~out["ats_no_pick"]
    )
    out["ats_hit"] = out["ats_gradable"] & (
        out["model_picks_home_cover"] == out["home_covered"]
    )
    return out


def rate(hits, n: int) -> float:
    return float(hits) / n if n else float("nan")


def fmt_pct(x: float) -> str:
    return "" if pd.isna(x) else f"{100 * x:5.1f}%"


def main() -> None:
    preds = load_oos_predictions()
    cal = cal_from_predictions(preds)
    scored = predicted_winner_table(preds, cal)
    scored["bucket"] = scored["pred_wp"].map(bucket_label)
    scored = attach_ats(scored, load_closing_spreads())

    kept = scored.loc[scored["bucket"] != EXCLUDE_BUCKET]
    ats_all = scored.loc[scored["ats_gradable"]]
    ats_kept = kept.loc[kept["ats_gradable"]]

    n_all = int(len(scored))
    n_kept = int(len(kept))
    n_excl = n_all - n_kept
    n_push_all = int(scored["ats_push"].sum())
    n_push_kept = int(kept["ats_push"].sum())

    su_all = rate(int(scored["predicted_winner_won"].sum()), n_all)
    su_kept = rate(int(kept["predicted_winner_won"].sum()), n_kept)
    ats_acc_all = rate(int(ats_all["ats_hit"].sum()), int(len(ats_all)))
    ats_acc_kept = rate(int(ats_kept["ats_hit"].sum()), int(len(ats_kept)))

    print("OOS backtest: straight-up vs against-the-spread")
    print(f"Standing model: recency + turnover_adjustment  |  OOS games: {n_all}")
    print("SU: predicted winner actually won (ties = miss).")
    print("ATS: model predicted point margin vs nflverse closing spread_line")
    print("     (positive = home favored). Pushes excluded from ATS denominator.")
    print(f"Filter: drop predicted-winner WP in the {EXCLUDE_BUCKET} bucket.")
    print(f"Pushes: {n_push_all} unfiltered, {n_push_kept} in the filtered set.")
    print()

    table = pd.DataFrame(
        {
            "metric": [
                "n games",
                "straight-up accuracy",
                "ATS accuracy",
                "ATS vs 52.4% breakeven",
            ],
            "unfiltered": [
                str(n_all),
                f"{fmt_pct(su_all)}   ({int(scored['predicted_winner_won'].sum())}/{n_all})",
                f"{fmt_pct(ats_acc_all)}   ({int(ats_all['ats_hit'].sum())}/{len(ats_all)})",
                ""
                if pd.isna(ats_acc_all)
                else f"{100 * (ats_acc_all - BREAKEVEN):+.1f} pp",
            ],
            "filtered_drop_50_60": [
                str(n_kept),
                f"{fmt_pct(su_kept)}   ({int(kept['predicted_winner_won'].sum())}/{n_kept})",
                f"{fmt_pct(ats_acc_kept)}   ({int(ats_kept['ats_hit'].sum())}/{len(ats_kept)})",
                ""
                if pd.isna(ats_acc_kept)
                else f"{100 * (ats_acc_kept - BREAKEVEN):+.1f} pp",
            ],
        }
    )
    print(table.to_string(index=False))
    print()
    print(
        f"Games excluded by {EXCLUDE_BUCKET} WP filter: {n_excl}  |  "
        f"ATS graded unfiltered {len(ats_all)}, filtered {len(ats_kept)}"
    )
    print(
        "52.4% is the ATS breakeven (vig), not a straight-up benchmark. "
        "Favorites win outright more than half the time by construction."
    )

    log_run(
        version_tag="v2_ats_backtest",
        layers_active="recency,turnover_adjustment,filter_50_60",
        games=int(len(ats_kept)),
        r=cal["r"],
        r_squared=cal["r_squared"],
        slope=cal["slope"],
        pick_accuracy_su=su_kept,
        pick_accuracy_ats=ats_acc_kept,
        notes=(
            f"ATS backtest (not SU). Closing nflverse spread_line, pushes excluded. "
            f"Filtered drop {EXCLUDE_BUCKET} WP: SU {100 * su_kept:.1f}% "
            f"({int(kept['predicted_winner_won'].sum())}/{n_kept}) vs earlier "
            f"v2_filtered_backtest SU 73.6%; ATS {100 * ats_acc_kept:.1f}% "
            f"({int(ats_kept['ats_hit'].sum())}/{len(ats_kept)}) vs 52.4% breakeven. "
            f"Unfiltered: SU {100 * su_all:.1f}% ({int(scored['predicted_winner_won'].sum())}/{n_all}), "
            f"ATS {100 * ats_acc_all:.1f}% ({int(ats_all['ats_hit'].sum())}/{len(ats_all)})."
        ),
    )

    print()
    print("metrics_log.csv")
    print(LOG_PATH.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
