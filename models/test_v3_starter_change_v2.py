"""Isolated OOS test: official injury-report starter_change (v2).

Replaces the snap-count proxy (v3_starter_change). Tested on top of the live
baseline (recency + turnover + QB-change). Blend/odds lever is not touched.
Not wired to the dashboard.
"""

from __future__ import annotations

from starter_change import (
    load_injuries,
    load_snap_counts,
    starter_change_features,
    starter_change_gap_epa,
    team_game_net_epa,
)
from test_levers import pick_rates, print_on_off, verdict
from metrics_log import log_run
from rating_v1 import load_pbp
from validate_oos import layers_active_label, run_walk_forward


def main() -> None:
    pbp = load_pbp()
    baseline_layers = layers_active_label(True, True, False, use_qb_change=True)

    print("Isolated lever test on CURRENT LIVE BASELINE")
    print("Baseline: recency + turnover_adjustment + qb_change")
    print("starter_change v2 ON TOP OF QB-change, not vs the pre-QB baseline.")
    print("Blend/odds lever not touched. Not wired to the dashboard.")
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
        f"OFF baseline (QB-change ON, starter-change OFF)  "
        f"r={off_fit.rvalue:.4f}  R^2={off_fit.rvalue**2:.4f}  "
        f"SU={100 * su_off:.1f}% ({n_off})  ATS={100 * ats_off:.1f}% ({n_ats_off})"
    )
    print()

    print("=" * 72)
    print("LEVER - Key starter Out/Doubtful (WR1 + CB1/top DB + starting OL)")
    injuries = load_injuries()
    snaps = load_snap_counts()
    feat = starter_change_features(pbp, snaps, injuries)
    n_home = int(feat["home_starter_change"].sum())
    n_away = int(feat["away_starter_change"].sum())
    n_ol = int((feat["home_ol_change"] | feat["away_ol_change"]).sum())
    n_wr = int((feat["home_wr_change"] | feat["away_wr_change"]).sum())
    n_cb = int((feat["home_cb_change"] | feat["away_cb_change"]).sum())
    net = team_game_net_epa(pbp)
    gap, gap_note = starter_change_gap_epa(net, feat, int(feat["week"].max()) + 1)
    seasons = feat.attrs.get("injury_seasons")
    print(
        "Data: nfl_data_py import_injuries() / nflverse weekly injury reports "
        f"(cached data/raw/injuries_2023_2025.csv, seasons={seasons})."
    )
    print("Final weekly report_status only. Flag = Out or Doubtful. Questionable ignored.")
    print("WR1/CB1/OL identity from the team's PRIOR game snap counts (pre-game, no leak).")
    print("WR1 = last game's highest offense-snap WR. CB1 = last game's highest")
    print("defense-snap CB (DB fallback if no CB row). OL = last game's top-5 C/G/T/OL;")
    print("OL flags if ANY of those five is Out/Doubtful. Combined OR flag.")
    print("QB excluded (already live on qb_change).")
    print(
        f"Flags weeks all: home={n_home} away={n_away}  "
        f"games with any OL/WR/CB flag={int(((feat['home_starter_change']==1)|(feat['away_starter_change']==1)).sum())}  "
        f"(OL-side {n_ol}, WR-side {n_wr}, CB-side {n_cb})  "
        f"CB-as-DB fallback team-games={feat.attrs.get('cb_db_fallback_team_games')}"
    )
    print(f"Full-season gap context (not used as-is in walk-forward): {gap:.4f}  {gap_note}")
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
        "starter_change_v2",
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
        "Replaces noisy snap-count version (v3_starter_change) with official injury "
        "report data (nflverse import_injuries, seasons 2023-2025). "
        "starter_change ON TOP OF QB-change (live baseline), not vs pre-QB baseline. "
        "Combined WR1/CB1/OL Out-or-Doubtful flag from final weekly report. "
        f"OFF r={off_fit.rvalue:.4f} R^2={off_fit.rvalue**2:.4f} "
        f"SU={100 * su_off:.1f}% ATS={100 * ats_off:.1f}%; "
        f"ON r={on_fit.rvalue:.4f} R^2={on_fit.rvalue**2:.4f} "
        f"SU={100 * su_on:.1f}% ATS={100 * ats_on:.1f}%; "
        f"verdict={starter_label}. Not wired to dashboard."
    )
    log_run(
        version_tag="v3_starter_change_v2",
        layers_active=f"{baseline_layers},starter_change",
        games=len(on_table),
        r=float(on_fit.rvalue),
        r_squared=float(on_fit.rvalue**2),
        slope=float(on_fit.slope),
        notes=notes,
        pick_accuracy_su=su_on,
        pick_accuracy_ats=ats_on,
    )
    print("Logged metrics_log.csv version_tag=v3_starter_change_v2")
    print()
    print("=" * 72)
    print("Summary (R^2 noise <0.005 and SU noise <1pp -> NO EFFECT; both must clear)")
    print(
        f"v3_starter_change_v2: {starter_label}  "
        f"(dR^2={delta_r2:+.4f}, dSU={100 * delta_su:+.1f}pp)"
    )
    print(f"VERDICT: {starter_label}")


if __name__ == "__main__":
    main()
