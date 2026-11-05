"""Isolated OOS tests: market blend weights and key starter-change.

Both tested on top of the live baseline (recency + turnover + QB-change).
Not wired to the dashboard.
"""

from __future__ import annotations

import json

import pandas as pd
from scipy.stats import linregress

from backtest_filtered import attach_ats, load_closing_spreads, rate
from calibration_check import predicted_winner_table
from metrics_log import log_run
from rating_v1 import ROOT, load_pbp
from starter_change import (
    load_snap_counts,
    starter_change_features,
    starter_change_gap_epa,
    team_game_net_epa,
)
from test_levers import print_on_off, verdict
from validate_oos import layers_active_label, run_walk_forward

CAL_PATH = ROOT / "data" / "oos_calibration.json"
BLEND_WEIGHTS = (
    (0.70, "v3_blend_70", "70/30 model/vegas"),
    (0.50, "v3_blend_50", "50/50 model/vegas"),
    (0.30, "v3_blend_30", "30/70 model/vegas"),
)
LINE_NOTE = (
    "VEGAS SOURCE: nflverse/nflfastR spread_line is the CLOSING line "
    "(Pro-Football-Reference), not a confirmed pre-kickoff/open line. "
    "No opening-line history is in this repo for 2025. Using close leaks "
    "information that would not be available as a pre-game blend and also "
    "makes ATS vs that same close a contaminated grade."
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


def apply_blend(table: pd.DataFrame, model_weight: float, stored: dict, spreads: pd.DataFrame) -> pd.DataFrame:
    """Blend model point margin with home-spread close; map back to EPA-scale predicted_diff."""
    out = table.merge(spreads, on="game_id", how="left")
    intercept = float(stored["intercept"])
    slope = float(stored["slope"])
    model_pts = intercept + slope * out["predicted_diff"]
    vegas_pts = out["spread_line"]
    blended_pts = model_pts.copy()
    has_line = vegas_pts.notna()
    blended_pts.loc[has_line] = (
        model_weight * model_pts.loc[has_line]
        + (1.0 - model_weight) * vegas_pts.loc[has_line]
    )
    out["predicted_diff"] = (blended_pts - intercept) / slope
    return out


def main() -> None:
    pbp = load_pbp()
    stored = json.loads(CAL_PATH.read_text(encoding="utf-8"))
    spreads = load_closing_spreads()
    baseline_layers = layers_active_label(True, True, False, use_qb_change=True)

    print("Isolated lever tests on CURRENT LIVE BASELINE")
    print("Baseline: recency + turnover_adjustment + qb_change")
    print("Each new lever is tested ON TOP OF QB-change, not vs the pre-QB baseline.")
    print("Not bundled with each other. Not wired to the dashboard.")
    print()
    print(LINE_NOTE)
    print()

    off_table, off_fit, _ = run_walk_forward(
        quiet=True,
        use_recency_weight=True,
        use_turnover_adjustment=True,
        use_qb_change=True,
        lever=None,
        pbp=pbp,
    )
    su_off, ats_off, n_off, n_ats_off = pick_rates(off_table)
    print(
        f"OFF baseline (QB-change ON, blend OFF, starter-change OFF)  "
        f"r={off_fit.rvalue:.4f}  R^2={off_fit.rvalue**2:.4f}  "
        f"SU={100 * su_off:.1f}% ({n_off})  ATS={100 * ats_off:.1f}% ({n_ats_off})"
    )
    print()

    blend_rows = []
    print("=" * 72)
    print("LEVER A - Market blend (three weights, not averaged)")
    print()
    for weight, tag, title in BLEND_WEIGHTS:
        on_table = apply_blend(off_table, weight, stored, spreads)
        on_table = on_table.drop(columns=["spread_line"], errors="ignore")
        on_fit = linregress(on_table["predicted_diff"], on_table["actual_diff"])
        su_on, ats_on, n_on, n_ats_on = pick_rates(on_table)
        print_on_off(
            title,
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
        blend_rows.append((tag, title, label, delta_r2, delta_su, on_fit, su_on, ats_on, n_on, n_ats_on))
        notes = (
            f"Market blend {title} ON TOP OF QB-change (live baseline), blend OFF is that baseline. "
            f"NOT tested vs pre-QB baseline. {LINE_NOTE} "
            f"OFF r={off_fit.rvalue:.4f} R^2={off_fit.rvalue**2:.4f} "
            f"SU={100 * su_off:.1f}% ATS={100 * ats_off:.1f}%; "
            f"ON r={on_fit.rvalue:.4f} R^2={on_fit.rvalue**2:.4f} "
            f"SU={100 * su_on:.1f}% ATS={100 * ats_on:.1f}%; "
            f"verdict={label}. Not wired to dashboard."
        )
        log_run(
            version_tag=tag,
            layers_active=f"{baseline_layers},blend_{int(100 * weight)}",
            games=len(on_table),
            r=float(on_fit.rvalue),
            r_squared=float(on_fit.rvalue**2),
            slope=float(on_fit.slope),
            notes=notes,
            pick_accuracy_su=su_on,
            pick_accuracy_ats=ats_on,
        )
        print(f"Logged metrics_log.csv version_tag={tag}  verdict={label}")
        print()

    best = max(blend_rows, key=lambda row: row[5].rvalue**2)
    print(
        f"Best blend weight by R^2: {best[0]} ({best[1]})  "
        f"verdict={best[2]}  (dR^2={best[3]:+.4f}, dSU={100 * best[4]:+.1f}pp)"
    )
    print()

    print("=" * 72)
    print("LEVER B - Key starter change (OL unit + WR1 + CB1, one combined flag)")
    snaps = load_snap_counts()
    feat = starter_change_features(pbp, snaps)
    n_home = int(feat["home_starter_change"].sum())
    n_away = int(feat["away_starter_change"].sum())
    n_ol = int((feat["home_ol_change"] | feat["away_ol_change"]).sum())
    n_wr = int((feat["home_wr_change"] | feat["away_wr_change"]).sum())
    n_cb = int((feat["home_cb_change"] | feat["away_cb_change"]).sum())
    net = team_game_net_epa(pbp)
    gap, gap_note = starter_change_gap_epa(net, feat, int(feat["week"].max()) + 1)
    print("Data: nflverse PFR snap counts (cached data/raw/snap_counts_2025.csv).")
    print("WR1 = most offense snaps among WR. CB1 = most defense snaps among CB")
    print("(DB fallback if a team-game has no CB row). OL = top 5 C/G/T/OL by snaps;")
    print("OL flags if overlap with last week's top 5 is < 4 (2+ of the 5 changed).")
    print("Week-to-week vs prior game, not a 3-game mode. In-game snaps, not pre-game IR.")
    print(
        f"Flags weeks all: home={n_home} away={n_away}  "
        f"games with any OL/WR/CB flag={int(((feat['home_starter_change']==1)|(feat['away_starter_change']==1)).sum())}  "
        f"(OL-side {n_ol}, WR-side {n_wr}, CB-side {n_cb})  "
        f"CB-as-DB fallback team-games={feat.attrs.get('cb_db_fallback_team_games')}"
    )
    print(f"Full-season gap context (not used as-is in walk-forward): {gap:.4f}  {gap_note}")
    print("Limitation: no official depth-chart 'starter' tag; snap-count proxies only.")
    print()

    on_table, on_fit, _ = run_walk_forward(
        quiet=True,
        use_recency_weight=True,
        use_turnover_adjustment=True,
        use_qb_change=True,
        lever="starter_change",
        pbp=pbp,
    )
    su_on, ats_on, n_on, n_ats_on = pick_rates(on_table)
    print_on_off(
        "starter_change",
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
    starter_label = verdict(delta_r2, delta_su)
    notes = (
        "starter_change ON TOP OF QB-change (live baseline), not vs pre-QB baseline. "
        "Combined OL/WR1/CB1 snap-count flag. "
        f"OFF r={off_fit.rvalue:.4f} R^2={off_fit.rvalue**2:.4f} "
        f"SU={100 * su_off:.1f}% ATS={100 * ats_off:.1f}%; "
        f"ON r={on_fit.rvalue:.4f} R^2={on_fit.rvalue**2:.4f} "
        f"SU={100 * su_on:.1f}% ATS={100 * ats_on:.1f}%; "
        f"verdict={starter_label}. Not wired to dashboard."
    )
    log_run(
        version_tag="v3_starter_change",
        layers_active=f"{baseline_layers},starter_change",
        games=len(on_table),
        r=float(on_fit.rvalue),
        r_squared=float(on_fit.rvalue**2),
        slope=float(on_fit.slope),
        notes=notes,
        pick_accuracy_su=su_on,
        pick_accuracy_ats=ats_on,
    )
    print(f"Logged metrics_log.csv version_tag=v3_starter_change")
    print()
    print("=" * 72)
    print("Summary (R^2 noise <0.005 and SU noise <1pp -> NO EFFECT)")
    print(
        f"{best[0]} (best blend by R^2): {best[2]}  "
        f"(dR^2={best[3]:+.4f}, dSU={100 * best[4]:+.1f}pp)"
    )
    print(
        f"v3_starter_change: {starter_label}  "
        f"(dR^2={delta_r2:+.4f}, dSU={100 * delta_su:+.1f}pp)"
    )


if __name__ == "__main__":
    main()
