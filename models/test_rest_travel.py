"""Walk-forward OOS test: rest/travel layer ON vs OFF (recency + turnover stay on)."""

from __future__ import annotations

import pandas as pd
from scipy.stats import linregress

from backtest_filtered import attach_ats, load_closing_spreads, rate
from calibration_check import predicted_winner_table
from metrics_log import log_run
from rating_v1 import load_pbp
from validate_oos import layers_active_label, run_walk_forward


def cal_from_predictions(preds: pd.DataFrame) -> dict:
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


def pick_rates(table: pd.DataFrame) -> tuple[float, float, int, int]:
    cal = cal_from_predictions(table)
    scored = predicted_winner_table(table, cal)
    scored = attach_ats(scored, load_closing_spreads())
    su = rate(int(scored["predicted_winner_won"].sum()), int(len(scored)))
    graded = scored.loc[scored["ats_gradable"]]
    ats = rate(int(graded["ats_hit"].sum()), int(len(graded)))
    return su, ats, int(len(scored)), int(len(graded))


def effect_word(delta: float, slack: float) -> str:
    if abs(delta) < slack:
        return "no effect"
    return "improved" if delta > 0 else "hurt"


def main() -> None:
    pbp = load_pbp()
    print("Rest/travel layer test — same walk-forward OOS as prior layers")
    print("Standing flags stay on: recency + turnover_adjustment")
    print("Rest: <=4 days. Travel: away team >=1500 miles since last game site.")
    print()

    off_table, off_fit, _ = run_walk_forward(
        quiet=True,
        use_recency_weight=True,
        use_turnover_adjustment=True,
        use_rest_travel=False,
        use_qb_change=False,
        pbp=pbp,
    )
    on_table, on_fit, _ = run_walk_forward(
        quiet=True,
        use_recency_weight=True,
        use_turnover_adjustment=True,
        use_rest_travel=True,
        use_qb_change=False,
        pbp=pbp,
    )

    su_off, ats_off, n_off, n_ats_off = pick_rates(off_table)
    su_on, ats_on, n_on, n_ats_on = pick_rates(on_table)

    comparison = pd.DataFrame(
        {
            "metric": ["n_games", "r", "r_squared", "pick_accuracy_su", "pick_accuracy_ats"],
            "OFF": [
                len(off_table),
                off_fit.rvalue,
                off_fit.rvalue**2,
                su_off,
                ats_off,
            ],
            "ON": [
                len(on_table),
                on_fit.rvalue,
                on_fit.rvalue**2,
                su_on,
                ats_on,
            ],
        }
    )
    comparison["delta"] = comparison["ON"] - comparison["OFF"]
    pd.set_option("display.width", 120)
    pd.set_option("display.float_format", lambda x: f"{x: .4f}")
    print("ON vs OFF (rest_travel)")
    print(comparison.to_string(index=False))
    print()
    print(
        f"OFF  r={off_fit.rvalue:.4f}  R^2={off_fit.rvalue**2:.4f}  "
        f"SU={100 * su_off:.1f}% ({n_off})  ATS={100 * ats_off:.1f}% ({n_ats_off})"
    )
    print(
        f"ON   r={on_fit.rvalue:.4f}  R^2={on_fit.rvalue**2:.4f}  "
        f"SU={100 * su_on:.1f}% ({n_on})  ATS={100 * ats_on:.1f}% ({n_ats_on})"
    )

    r_effect = effect_word(float(on_fit.rvalue - off_fit.rvalue), 0.002)
    su_effect = effect_word(su_on - su_off, 0.005)
    ats_effect = effect_word(ats_on - ats_off, 0.005)
    if r_effect == su_effect == ats_effect:
        overall = r_effect
    else:
        overall = (
            f"mixed (r {r_effect}, SU {su_effect}, ATS {ats_effect})"
        )
    print(f"OOS accuracy vs standing model: {overall}")

    notes = (
        f"rest_travel ON vs OFF (recency+turnover stay on). "
        f"OFF r={off_fit.rvalue:.4f} R^2={off_fit.rvalue**2:.4f} "
        f"SU={100 * su_off:.1f}% ATS={100 * ats_off:.1f}%; "
        f"ON r={on_fit.rvalue:.4f} R^2={on_fit.rvalue**2:.4f} "
        f"SU={100 * su_on:.1f}% ATS={100 * ats_on:.1f}%; "
        f"effect: {overall}. Not wired to dashboard."
    )
    log_run(
        version_tag="v3_rest_travel",
        layers_active=layers_active_label(True, True, True),
        games=len(on_table),
        r=float(on_fit.rvalue),
        r_squared=float(on_fit.rvalue**2),
        slope=float(on_fit.slope),
        notes=notes,
        pick_accuracy_su=su_on,
        pick_accuracy_ats=ats_on,
    )
    print("Logged metrics_log.csv version_tag=v3_rest_travel")


if __name__ == "__main__":
    main()
