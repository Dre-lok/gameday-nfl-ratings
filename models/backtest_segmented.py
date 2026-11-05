"""Slice existing ATS OOS backtest by edge size and key-number spreads."""

from __future__ import annotations

import pandas as pd

from backtest_filtered import (
    BREAKEVEN,
    attach_ats,
    cal_from_predictions,
    load_closing_spreads,
    rate,
)
from calibration_check import load_oos_predictions, predicted_winner_table
from metrics_log import log_run

EDGE_BUCKETS = [
    (0.0, 1.0, "0-1 pt"),
    (1.0, 2.0, "1-2 pt"),
    (2.0, 3.0, "2-3 pt"),
    (3.0, 5.0, "3-5 pt"),
    (5.0, float("inf"), "5+ pt"),
]
KEY_NUMBERS = (3.0, 7.0)
KEY_TOLERANCE = 0.5


def edge_bucket_label(edge_size: float) -> str:
    for low, high, label in EDGE_BUCKETS:
        if low <= edge_size < high:
            return label
    return "5+ pt"


def near_key_number(spread_line: float) -> bool:
    """True if |market spread| is on or within 0.5 of 3 or 7."""
    mag = abs(float(spread_line))
    return any(abs(mag - key) <= KEY_TOLERANCE for key in KEY_NUMBERS)


def ats_slice_row(chunk: pd.DataFrame, label: str) -> dict:
    n = int(len(chunk))
    hits = int(chunk["ats_hit"].sum()) if n else 0
    acc = rate(hits, n)
    gap = acc - BREAKEVEN if n else float("nan")
    return {
        "bucket": label,
        "n": n,
        "ats_hits": hits,
        "ats_accuracy": acc,
        "vs_breakeven_pp": 100 * gap if n else float("nan"),
    }


def format_segment_table(rows: list[dict]) -> pd.DataFrame:
    table = pd.DataFrame(rows)
    show = table.copy()
    show["ats_accuracy"] = show.apply(
        lambda r: ""
        if pd.isna(r["ats_accuracy"])
        else f"{100 * r['ats_accuracy']:5.1f}%   ({int(r['ats_hits'])}/{int(r['n'])})",
        axis=1,
    )
    show["vs_52.4%"] = show["vs_breakeven_pp"].map(
        lambda x: "" if pd.isna(x) else f"{x:+.1f} pp"
    )
    return show[["bucket", "n", "ats_accuracy", "vs_52.4%"]]


def notes_from_rows(rows: list[dict]) -> str:
    parts = []
    for row in rows:
        if row["n"] == 0 or pd.isna(row["ats_accuracy"]):
            parts.append(f"{row['bucket']}: n=0")
            continue
        parts.append(
            f"{row['bucket']}: n={row['n']} ATS={100 * row['ats_accuracy']:.1f}% "
            f"({100 * (row['ats_accuracy'] - BREAKEVEN):+.1f} pp vs 52.4%)"
        )
    return "; ".join(parts)


def load_ats_graded() -> tuple[pd.DataFrame, dict]:
    preds = load_oos_predictions()
    cal = cal_from_predictions(preds)
    scored = predicted_winner_table(preds, cal)
    scored = attach_ats(scored, load_closing_spreads())
    graded = scored.loc[scored["ats_gradable"]].copy()
    # Same units as ATS grade: pred_margin and spread_line are both home-favored points.
    graded["edge_size"] = (graded["pred_margin"] - graded["spread_line"]).abs()
    graded["edge_bucket"] = graded["edge_size"].map(edge_bucket_label)
    graded["near_key"] = graded["spread_line"].map(near_key_number)
    return graded, cal


def main() -> None:
    graded, cal = load_ats_graded()

    print("ATS edge-size segments (unfiltered graded OOS, no 50-60% WP filter)")
    print(f"n = {len(graded)}  |  edge_size = |predicted point margin - market spread_line|")
    print("spread_line is nflverse close (positive = home favored).")
    print()

    edge_rows = []
    for _, _, label in EDGE_BUCKETS:
        chunk = graded.loc[graded["edge_bucket"] == label]
        edge_rows.append(ats_slice_row(chunk, label))
    print(format_segment_table(edge_rows).to_string(index=False))

    print()
    print("ATS key-number segments")
    print("Near a key number: |market spread| on or within 0.5 of 3 or 7 (e.g. 2.5, 3, 3.5, 6.5, 7, 7.5).")
    print()
    key_rows = [
        ats_slice_row(graded.loc[graded["near_key"]], "near 3 or 7"),
        ats_slice_row(graded.loc[~graded["near_key"]], "not near 3 or 7"),
    ]
    print(format_segment_table(key_rows).to_string(index=False))

    overall_ats = rate(int(graded["ats_hit"].sum()), int(len(graded)))
    log_run(
        version_tag="v2_edge_segments",
        layers_active="recency,turnover_adjustment",
        games=int(len(graded)),
        r=cal["r"],
        r_squared=cal["r_squared"],
        slope=cal["slope"],
        pick_accuracy_ats=overall_ats,
        notes="ATS by |pred margin - spread_line|: " + notes_from_rows(edge_rows),
    )
    near = key_rows[0]
    log_run(
        version_tag="v2_key_number_segments",
        layers_active="recency,turnover_adjustment",
        games=int(near["n"]),
        r=cal["r"],
        r_squared=cal["r_squared"],
        slope=cal["slope"],
        pick_accuracy_ats=near["ats_accuracy"],
        notes="ATS near key numbers (3/7 ±0.5) vs rest: " + notes_from_rows(key_rows),
    )


SMALL_SAMPLE_N = 40
COMBINED_EDGE_MIN = 3.0


def combined_filter() -> None:
    """Keep large-edge, non-key-number games; sanity-check the inverse."""
    graded, cal = load_ats_graded()
    n_all = int(len(graded))
    large_edge = graded["edge_size"] >= COMBINED_EDGE_MIN
    near_key = graded["near_key"]

    keep = large_edge & ~near_key
    edge_too_small_only = ~large_edge & ~near_key
    near_key_only = large_edge & near_key
    both = ~large_edge & near_key
    inverse = both

    kept_row = ats_slice_row(graded.loc[keep], "edge >= 3 AND not near 3/7")
    inverse_row = ats_slice_row(graded.loc[inverse], "edge < 3 AND near 3/7")

    n_kept = kept_row["n"]
    n_small_only = int(edge_too_small_only.sum())
    n_key_only = int(near_key_only.sum())
    n_both = int(both.sum())

    print("Combined ATS filter (unfiltered graded OOS)")
    print(f"Pool: {n_all} graded games")
    print("KEEP: edge_size >= 3  AND  not near a key number (3/7 within 0.5)")
    print()
    print(f"Games kept:     {n_kept}")
    print(
        f"ATS accuracy:   {100 * kept_row['ats_accuracy']:.1f}%   "
        f"({kept_row['ats_hits']}/{n_kept})"
        if n_kept
        else "ATS accuracy:   n/a"
    )
    print(
        f"vs 52.4%:       {kept_row['vs_breakeven_pp']:+.1f} pp"
        if n_kept
        else "vs 52.4%:       n/a"
    )
    if n_kept < SMALL_SAMPLE_N:
        print()
        print(
            f"FLAG: n={n_kept} is below ~{SMALL_SAMPLE_N}. "
            "Treat as promising, needs more seasons to confirm — not proven."
        )
    print()
    print("Excluded (partition of the rest):")
    print(f"  edge too small only (edge < 3, not near key): {n_small_only}")
    print(f"  near a key number only (edge >= 3, near 3/7): {n_key_only}")
    print(f"  both (edge < 3 AND near 3/7):                 {n_both}")
    print(f"  excluded total:                               {n_all - n_kept}")

    print()
    print("Inverse (worst combination): edge < 3 AND near a key number")
    print(f"Games:          {inverse_row['n']}")
    print(
        f"ATS accuracy:   {100 * inverse_row['ats_accuracy']:.1f}%   "
        f"({inverse_row['ats_hits']}/{inverse_row['n']})"
        if inverse_row["n"]
        else "ATS accuracy:   n/a"
    )
    print(
        f"vs 52.4%:       {inverse_row['vs_breakeven_pp']:+.1f} pp"
        if inverse_row["n"]
        else "vs 52.4%:       n/a"
    )

    log_run(
        version_tag="v2_combined_filter",
        layers_active="recency,turnover_adjustment,edge_ge_3,not_near_key",
        games=n_kept,
        r=cal["r"],
        r_squared=cal["r_squared"],
        slope=cal["slope"],
        pick_accuracy_ats=kept_row["ats_accuracy"],
        notes=(
            f"KEEP edge>=3 and not near 3/7: n={n_kept}/{n_all} "
            f"ATS={100 * kept_row['ats_accuracy']:.1f}% "
            f"({kept_row['ats_hits']}/{n_kept}, "
            f"{kept_row['vs_breakeven_pp']:+.1f} pp vs 52.4%)"
            + (
                f"; FLAG n<{SMALL_SAMPLE_N} treat as unproven"
                if n_kept < SMALL_SAMPLE_N
                else ""
            )
            + f"; excluded edge-small-only={n_small_only}, "
            f"key-only={n_key_only}, both={n_both}; "
            f"inverse edge<3 AND near key: n={inverse_row['n']} "
            f"ATS={100 * inverse_row['ats_accuracy']:.1f}%"
        ),
    )


E5_COMBINED_MIN = 5.0
E3_COMBINED_ATS = 0.565
E3_COMBINED_N = 46


def combined_filter_e5() -> None:
    """Tighter combined filter: edge >= 5 and not near a key number."""
    graded, cal = load_ats_graded()
    n_all = int(len(graded))
    large_edge = graded["edge_size"] >= E5_COMBINED_MIN
    near_key = graded["near_key"]
    keep = large_edge & ~near_key
    kept_row = ats_slice_row(graded.loc[keep], "edge >= 5 AND not near 3/7")
    e5_alone = ats_slice_row(graded.loc[large_edge], "edge >= 5 (alone)")

    n_kept = kept_row["n"]
    acc = kept_row["ats_accuracy"]

    print("Combined ATS filter (tighter)")
    print(f"Pool: {n_all} graded games")
    print("KEEP: edge_size >= 5.0  AND  not near a key number (3/7 within 0.5)")
    print()
    print(f"Games kept:     {n_kept}")
    print(
        f"ATS accuracy:   {100 * acc:.1f}%   ({kept_row['ats_hits']}/{n_kept})"
        if n_kept
        else "ATS accuracy:   n/a"
    )
    print(
        f"vs 52.4%:       {kept_row['vs_breakeven_pp']:+.1f} pp"
        if n_kept
        else "vs 52.4%:       n/a"
    )
    if n_kept < SMALL_SAMPLE_N:
        print()
        print("SMALL SAMPLE - NEEDS MORE SEASONS")

    e5_alone_pct = 100 * e5_alone["ats_accuracy"] if e5_alone["n"] else float("nan")
    print()
    print(
        f"vs edge>=3 combined ({100 * E3_COMBINED_ATS:.1f}%, n={E3_COMBINED_N}) "
        f"and vs edge>=5 alone ({e5_alone_pct:.1f}%, n={e5_alone['n']}): "
        f"this filter {100 * acc:.1f}% (n={n_kept})"
        if n_kept
        else "comparison: n/a"
    )

    log_run(
        version_tag="v2_combined_filter_e5",
        layers_active="recency,turnover_adjustment,edge_ge_5,not_near_key",
        games=n_kept,
        r=cal["r"],
        r_squared=cal["r_squared"],
        slope=cal["slope"],
        pick_accuracy_ats=acc,
        notes=(
            f"KEEP edge>=5 and not near 3/7: n={n_kept}/{n_all} "
            f"ATS={100 * acc:.1f}% ({kept_row['ats_hits']}/{n_kept}, "
            f"{kept_row['vs_breakeven_pp']:+.1f} pp vs 52.4%)"
            + (
                "; SMALL SAMPLE - NEEDS MORE SEASONS"
                if n_kept < SMALL_SAMPLE_N
                else ""
            )
            + f"; vs e3 combined {100 * E3_COMBINED_ATS:.1f}% n={E3_COMBINED_N}; "
            f"vs e5-alone {e5_alone_pct:.1f}% n={e5_alone['n']}"
        ),
    )


SEASON2_YEAR = 2024
SEASON3_YEAR = 2023
OLD_EDGE5_N = 46
OLD_EDGE5_HITS = 28  # 60.9% of 46
SEASON2_EDGE5_N = 50
SEASON2_EDGE5_HITS = 23  # 46.0% of 50
HOLDS_ATS_MIN = 0.55
HOLDS_N_MIN = 100


def _walk_forward_edge5(year: int):
    """Full pipeline on one season; return cal + edge>=5 ATS slice."""
    import sys

    from rating_v1 import ROOT
    from validate_oos import run_walk_forward

    sys.path.insert(0, str(ROOT / "data"))
    from pull_data import pull_season

    path = pull_season(year)
    pbp = pd.read_csv(path, low_memory=False)
    table, _, _ = run_walk_forward(quiet=True, pbp=pbp)
    cal = cal_from_predictions(table)
    scored = predicted_winner_table(table, cal)
    spreads = (
        pbp.dropna(subset=["game_id", "spread_line"])
        .drop_duplicates("game_id")[["game_id", "spread_line"]]
        .copy()
    )
    scored = attach_ats(scored, spreads)
    graded = scored.loc[scored["ats_gradable"]].copy()
    graded["edge_size"] = (graded["pred_margin"] - graded["spread_line"]).abs()
    kept = graded.loc[graded["edge_size"] >= 5.0]
    row = ats_slice_row(kept, f"{year} edge>=5")
    return table, cal, graded, row


def confirm_edge5_season2(year: int = SEASON2_YEAR) -> None:
    """Walk-forward a held-out season and grade the edge>=5 ATS bucket only."""
    import sys

    from rating_v1 import ROOT
    from validate_oos import run_walk_forward

    sys.path.insert(0, str(ROOT / "data"))
    from pull_data import pull_season

    path = pull_season(year)
    pbp = pd.read_csv(path, low_memory=False)
    table, _, _ = run_walk_forward(quiet=True, pbp=pbp)
    cal = cal_from_predictions(table)
    scored = predicted_winner_table(table, cal)
    spreads = (
        pbp.dropna(subset=["game_id", "spread_line"])
        .drop_duplicates("game_id")[["game_id", "spread_line"]]
        .copy()
    )
    scored = attach_ats(scored, spreads)
    graded = scored.loc[scored["ats_gradable"]].copy()
    graded["edge_size"] = (graded["pred_margin"] - graded["spread_line"]).abs()
    kept = graded.loc[graded["edge_size"] >= 5.0]
    row = ats_slice_row(kept, f"{year} edge>=5")

    n_new = row["n"]
    hits_new = row["ats_hits"]
    ats_new = row["ats_accuracy"]
    n_merged = OLD_EDGE5_N + n_new
    hits_merged = OLD_EDGE5_HITS + hits_new
    ats_merged = hits_merged / n_merged if n_merged else float("nan")

    print(f"edge>=5 confirmation: {year} walk-forward (never in the 2025 pool)")
    print(f"Pipeline: garbage-time, SRS, recency + turnover-luck, weeks 5+ OOS.")
    print(f"{year} graded ATS games: {len(graded)}  |  OOS predicted games: {len(table)}")
    print()
    print(f"n games (edge>=5):     {n_new}")
    print(
        f"ATS accuracy ({year}):  {100 * ats_new:.1f}%   ({hits_new}/{n_new})"
        if n_new
        else f"ATS accuracy ({year}):  n/a"
    )
    print(
        f"vs 52.4%:              {row['vs_breakeven_pp']:+.1f} pp"
        if n_new
        else "vs 52.4%:              n/a"
    )
    print()
    print(
        f"Merged with 2025 pool ({OLD_EDGE5_N} games, 60.9%): "
        f"n={n_merged}  ATS={100 * ats_merged:.1f}%   ({hits_merged}/{n_merged})  "
        f"{100 * (ats_merged - BREAKEVEN):+.1f} pp vs 52.4%"
    )

    log_run(
        version_tag="v2_edge5_outofsample_season2",
        layers_active="recency,turnover_adjustment,edge_ge_5",
        games=n_new,
        r=cal["r"],
        r_squared=cal["r_squared"],
        slope=cal["slope"],
        pick_accuracy_ats=ats_new,
        notes=(
            f"{year} OOS edge>=5 only (no key-number filter): n={n_new} "
            f"ATS={100 * ats_new:.1f}% ({hits_new}/{n_new}, "
            f"{row['vs_breakeven_pp']:+.1f} pp vs 52.4%); "
            f"merged with 2025 edge>=5 ({OLD_EDGE5_N} games, 60.9% / {OLD_EDGE5_HITS} hits): "
            f"n={n_merged} ATS={100 * ats_merged:.1f}% ({hits_merged}/{n_merged})"
        ),
    )


def confirm_edge5_season3(year: int = SEASON3_YEAR) -> None:
    """Third-season edge>=5 check; merge 2023+2024+2025 and close the research line."""
    table, cal, graded, row = _walk_forward_edge5(year)
    n_new = row["n"]
    hits_new = row["ats_hits"]
    ats_new = row["ats_accuracy"]

    n_merged = OLD_EDGE5_N + SEASON2_EDGE5_N + n_new
    hits_merged = OLD_EDGE5_HITS + SEASON2_EDGE5_HITS + hits_new
    ats_merged = hits_merged / n_merged if n_merged else float("nan")
    holds = (not pd.isna(ats_merged)) and ats_merged >= HOLDS_ATS_MIN and n_merged >= HOLDS_N_MIN
    status = "holds" if holds else "does not hold - treat as inconclusive"

    print(f"edge>=5 confirmation: {year} walk-forward (not in 2024/2025 pools)")
    print("Pipeline: garbage-time, SRS, recency + turnover-luck, weeks 5+ OOS.")
    print(f"{year} graded ATS games: {len(graded)}  |  OOS predicted games: {len(table)}")
    print()
    print(f"n games (edge>=5):     {n_new}")
    print(
        f"ATS accuracy ({year}):  {100 * ats_new:.1f}%   ({hits_new}/{n_new})"
        if n_new
        else f"ATS accuracy ({year}):  n/a"
    )
    print(
        f"vs 52.4%:              {row['vs_breakeven_pp']:+.1f} pp"
        if n_new
        else "vs 52.4%:              n/a"
    )
    print()
    print(
        f"3-season merged (2025 n={OLD_EDGE5_N} 60.9% + 2024 n={SEASON2_EDGE5_N} 46.0% + {year}): "
        f"n={n_merged}  ATS={100 * ats_merged:.1f}%   ({hits_merged}/{n_merged})  "
        f"{100 * (ats_merged - BREAKEVEN):+.1f} pp vs 52.4%"
    )
    print()
    print(f"EDGE RESEARCH STATUS: {status}")

    log_run(
        version_tag="v2_edge5_season3_pool",
        layers_active="recency,turnover_adjustment,edge_ge_5",
        games=n_merged,
        r=cal["r"],
        r_squared=cal["r_squared"],
        slope=cal["slope"],
        pick_accuracy_ats=ats_merged,
        notes=(
            f"{year} OOS edge>=5: n={n_new} ATS={100 * ats_new:.1f}% "
            f"({hits_new}/{n_new}, {row['vs_breakeven_pp']:+.1f} pp vs 52.4%); "
            f"3-season merged n={n_merged} ATS={100 * ats_merged:.1f}% "
            f"({hits_merged}/{n_merged}, {100 * (ats_merged - BREAKEVEN):+.1f} pp vs 52.4%); "
            f"EDGE RESEARCH STATUS: {status}"
        ),
    )


if __name__ == "__main__":
    main()
