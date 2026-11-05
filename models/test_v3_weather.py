"""Isolated OOS test: Open-Meteo weather lever (v3_weather).

Tested on top of the live baseline (recency + turnover + QB-change).
Does not touch starter-change, blend/odds, or the live QB-change lever.
Not wired to the dashboard.
"""

from __future__ import annotations

from metrics_log import log_run
from rating_v1 import load_pbp
from test_levers import pick_rates, print_on_off, verdict
from validate_oos import layers_active_label, run_walk_forward
from weather import pull_weather, weather_features


def main() -> None:
    pbp = load_pbp()
    baseline_layers = layers_active_label(True, True, False, use_qb_change=True)

    print("Isolated lever test on CURRENT LIVE BASELINE")
    print("Baseline: recency + turnover_adjustment + qb_change")
    print("weather ON TOP OF QB-change, not vs the pre-QB baseline.")
    print("starter-change and blend/odds levers not touched. Not wired to the dashboard.")
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
        f"OFF baseline (QB-change ON, weather OFF)  "
        f"r={off_fit.rvalue:.4f}  R^2={off_fit.rvalue**2:.4f}  "
        f"SU={100 * su_off:.1f}% ({n_off})  ATS={100 * ats_off:.1f}% ({n_ats_off})"
    )
    print()

    print("=" * 72)
    print("LEVER - Adverse weather (wind>=15mph OR precip present OR temp<=32F)")
    weather = pull_weather()
    feat = weather_features(pbp, weather)
    n_indoor = int(feat["indoor"].sum())
    n_flag = int(feat["weather_adverse"].sum())
    n_wind = int(feat["wind_flag"].sum())
    n_precip = int(feat["precip_flag"].sum())
    n_cold = int(feat["cold_flag"].sum())
    n_missing = int((feat["source"] == "missing_venue_or_kickoff").sum())
    print(
        "Data: Open-Meteo Historical Weather API (archive-api.open-meteo.com), "
        "no key, cached data/raw/weather_2023_2025.csv for seasons 2023-2025."
    )
    print("Kickoff hour at stadium lat/lon. Indoor/dome/closed roof = no weather effect, not pulled.")
    print("Retractable 'open' and outdoors are pulled. 2025 international sites overridden")
    print("to the actual venue when nflverse still lists the US home stadium.")
    print(
        f"2025 pbp games: n={len(feat)}  indoor/no-effect={n_indoor}  "
        f"adverse flag={n_flag}  (wind {n_wind}, precip {n_precip}, cold {n_cold})  "
        f"missing venue/kickoff={n_missing}"
    )
    print(
        f"Pool cache 2023-2025: n={len(weather)}  indoor={int(weather['indoor'].sum())}  "
        f"adverse={int(weather['weather_adverse'].sum())}"
    )
    print()

    on_table, on_fit, _ = run_walk_forward(
        quiet=True,
        use_recency_weight=True,
        use_turnover_adjustment=True,
        use_qb_change=True,
        lever="weather",
        pbp=pbp,
    )
    su_on, ats_on, n_on, n_ats_on = pick_rates(on_table)
    print_on_off(
        "weather",
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
    notes = (
        "Open-Meteo free historical data, no paid source needed. "
        "weather ON TOP OF QB-change (live baseline), not vs pre-QB baseline. "
        "Single combined flag: wind>=15mph OR precip>0 OR temp<=32F at kickoff; "
        "indoor/dome/closed roof = no weather effect. "
        f"OFF r={off_fit.rvalue:.4f} R^2={off_fit.rvalue**2:.4f} "
        f"SU={100 * su_off:.1f}% ATS={100 * ats_off:.1f}%; "
        f"ON r={on_fit.rvalue:.4f} R^2={on_fit.rvalue**2:.4f} "
        f"SU={100 * su_on:.1f}% ATS={100 * ats_on:.1f}%; "
        f"verdict={label}. Not wired to dashboard."
    )
    log_run(
        version_tag="v3_weather",
        layers_active=f"{baseline_layers},weather",
        games=len(on_table),
        r=float(on_fit.rvalue),
        r_squared=float(on_fit.rvalue**2),
        slope=float(on_fit.slope),
        notes=notes,
        pick_accuracy_su=su_on,
        pick_accuracy_ats=ats_on,
    )
    print("Logged metrics_log.csv version_tag=v3_weather")
    print()
    print("=" * 72)
    print("Summary (R^2 noise <0.005 and SU noise <1pp -> NO EFFECT; both must clear)")
    print(f"v3_weather: {label}  (dR^2={delta_r2:+.4f}, dSU={100 * delta_su:+.1f}pp)")
    print(f"VERDICT: {label}")


if __name__ == "__main__":
    main()
