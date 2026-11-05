"""Walk-forward OOS tests for three isolated levers (one flag at a time)."""

from __future__ import annotations

import pandas as pd
from scipy.stats import linregress

from backtest_filtered import attach_ats, load_closing_spreads, rate
from calibration_check import predicted_winner_table
from levers import backup_starter_gap_epa, qb_change_features, team_game_off_epa
from metrics_log import log_run
from rating_v1 import load_pbp
from validate_oos import layers_active_label, run_walk_forward

LEVERS = (
    ("qb_change", "v3_qb_change", "Starting QB change detection"),
    ("rest_continuous", "v3_rest_continuous", "Continuous rest (days since last game)"),
    ("travel_refined", "v3_travel_refined", "Travel + TZ shift + Denver altitude"),
)


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


def verdict(delta_r2: float, delta_su: float) -> str:
    """HELPED / HURT only if both R² and SU moved past noise; else NO EFFECT."""
    r2_noise = abs(delta_r2) < 0.005
    su_noise = abs(delta_su) < 0.01
    if r2_noise and su_noise:
        return "NO EFFECT"
    r2_up = delta_r2 >= 0.005
    r2_down = delta_r2 <= -0.005
    su_up = delta_su >= 0.01
    su_down = delta_su <= -0.01
    if r2_up and su_up:
        return "HELPED"
    if r2_down and su_down:
        return "HURT"
    return "NO EFFECT"


def print_on_off(name: str, off_fit, on_fit, su_off, ats_off, n_off, n_ats_off, su_on, ats_on, n_on, n_ats_on, off_n, on_n) -> None:
    comparison = pd.DataFrame(
        {
            "metric": ["n_games", "r", "r_squared", "pick_accuracy_su", "pick_accuracy_ats"],
            "OFF": [off_n, off_fit.rvalue, off_fit.rvalue**2, su_off, ats_off],
            "ON": [on_n, on_fit.rvalue, on_fit.rvalue**2, su_on, ats_on],
        }
    )
    comparison["delta"] = comparison["ON"] - comparison["OFF"]
    pd.set_option("display.width", 120)
    pd.set_option("display.float_format", lambda x: f"{x: .4f}")
    print(f"ON vs OFF ({name})")
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


def main() -> None:
    pbp = load_pbp()
    print("Isolated lever tests - same walk-forward OOS as prior layers")
    print("Standing flags stay on: recency + turnover_adjustment")
    print("One lever per run. Not combined. Not wired to dashboard.")
    print()

    qb_feat = qb_change_features(pbp)
    gap, gap_note = backup_starter_gap_epa(
        team_game_off_epa(pbp), qb_feat, int(qb_feat["week"].max()) + 1
    )
    n_home = int(qb_feat["home_qb_change"].sum())
    n_away = int(qb_feat["away_qb_change"].sum())
    print(
        f"QB change (full season, for context): home flags={n_home}  "
        f"away flags={n_away}  gap used when sample is thick={gap:.4f}  {gap_note}"
    )
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
    su_off, ats_off, n_off, n_ats_off = pick_rates(off_table)
    print(
        f"OFF baseline  r={off_fit.rvalue:.4f}  R^2={off_fit.rvalue**2:.4f}  "
        f"SU={100 * su_off:.1f}% ({n_off})  ATS={100 * ats_off:.1f}% ({n_ats_off})"
    )
    print()

    summaries = []
    for lever, tag, title in LEVERS:
        print("=" * 72)
        print(title)
        print(f"lever={lever}  version_tag={tag}")
        print()
        on_table, on_fit, _ = run_walk_forward(
            quiet=True,
            use_recency_weight=True,
            use_turnover_adjustment=True,
            use_rest_travel=False,
            lever=lever,
            use_qb_change=False,
            pbp=pbp,
        )
        su_on, ats_on, n_on, n_ats_on = pick_rates(on_table)
        print_on_off(
            lever,
            off_fit,
            on_fit,
            su_off,
            ats_off,
            n_off,
            n_ats_off,
            su_on,
            ats_on,
            n_on,
            n_ats_on,
            len(off_table),
            len(on_table),
        )
        delta_r2 = float(on_fit.rvalue**2 - off_fit.rvalue**2)
        delta_su = float(su_on - su_off)
        label = verdict(delta_r2, delta_su)
        summaries.append((tag, label, delta_r2, delta_su))
        notes = (
            f"{lever} ON vs OFF (recency+turnover stay on; isolated, not bundled). "
            f"OFF r={off_fit.rvalue:.4f} R^2={off_fit.rvalue**2:.4f} "
            f"SU={100 * su_off:.1f}% ATS={100 * ats_off:.1f}%; "
            f"ON r={on_fit.rvalue:.4f} R^2={on_fit.rvalue**2:.4f} "
            f"SU={100 * su_on:.1f}% ATS={100 * ats_on:.1f}%; "
            f"verdict={label}. Not wired to dashboard."
        )
        log_run(
            version_tag=tag,
            layers_active=layers_active_label(True, True, False, lever, use_qb_change=(lever == "qb_change")),
            games=len(on_table),
            r=float(on_fit.rvalue),
            r_squared=float(on_fit.rvalue**2),
            slope=float(on_fit.slope),
            notes=notes,
            pick_accuracy_su=su_on,
            pick_accuracy_ats=ats_on,
        )
        print(f"Logged metrics_log.csv version_tag={tag}")
        print()

    print("=" * 72)
    print("Summary (R^2 noise <0.005 and SU noise <1pp -> NO EFFECT)")
    for tag, label, delta_r2, delta_su in summaries:
        print(
            f"{tag}: {label}  "
            f"(ΔR²={delta_r2:+.4f}, ΔSU={100 * delta_su:+.1f}pp)"
        )


if __name__ == "__main__":
    main()
