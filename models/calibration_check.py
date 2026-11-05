"""Calibration of OOS win probabilities vs actual predicted-winner win rate."""

from __future__ import annotations

import json

import pandas as pd
from scipy.stats import norm

from rating_v1 import ROOT
from validate_oos import OOS_PREDICTIONS_PATH, run_walk_forward

CALIBRATION_PATH = ROOT / "data" / "oos_calibration.json"

BUCKETS = [
    (0.50, 0.60, "50-60%"),
    (0.60, 0.70, "60-70%"),
    (0.70, 0.80, "70-80%"),
    (0.80, 0.90, "80-90%"),
    (0.90, 1.01, "90%+"),
]


def load_oos_predictions() -> pd.DataFrame:
    if OOS_PREDICTIONS_PATH.exists():
        return pd.read_csv(OOS_PREDICTIONS_PATH)
    print("oos_predictions.csv not found; running walk-forward with current defaults...")
    table, _, _ = run_walk_forward(quiet=True)
    OOS_PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OOS_PREDICTIONS_PATH, index=False)
    return table


def predicted_winner_table(preds: pd.DataFrame, cal: dict) -> pd.DataFrame:
    out = preds.copy()
    out["pred_margin"] = cal["intercept"] + cal["slope"] * out["predicted_diff"]
    out["p_home"] = 1.0 - norm.cdf(0.5, loc=out["pred_margin"], scale=cal["residual_std"])
    out["predicted_home"] = out["p_home"] >= 0.5
    out["pred_wp"] = out["p_home"].where(out["predicted_home"], 1.0 - out["p_home"])
    home_won = out["actual_diff"] > 0
    away_won = out["actual_diff"] < 0
    out["predicted_winner_won"] = (out["predicted_home"] & home_won) | (
        (~out["predicted_home"]) & away_won
    )
    return out


def bucket_label(wp: float) -> str | None:
    for low, high, label in BUCKETS:
        if low <= wp < high:
            return label
    return None


def calibration_table(preds: pd.DataFrame, cal: dict) -> pd.DataFrame:
    scored = predicted_winner_table(preds, cal)
    scored["bucket"] = scored["pred_wp"].map(bucket_label)
    rows = []
    for _, _, label in BUCKETS:
        chunk = scored.loc[scored["bucket"] == label]
        n = int(len(chunk))
        mean_pred = float(chunk["pred_wp"].mean()) if n else float("nan")
        actual = float(chunk["predicted_winner_won"].mean()) if n else float("nan")
        rows.append(
            {
                "bucket": label,
                "n": n,
                "mean_predicted_wp": mean_pred,
                "actual_win_rate": actual,
                "gap": actual - mean_pred if n else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    preds = load_oos_predictions()
    cal = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
    table = calibration_table(preds, cal)

    pd.set_option("display.width", 120)
    pd.set_option("display.float_format", lambda x: f"{x: .4f}")

    print("Win-probability calibration (OOS predicted winner)")
    print("P(win) from Normal(calibrated margin, residual_std), P(margin > 0.5).")
    print("Bucket is the predicted winner's win probability; actual is how often they won.")
    print(f"Games: {len(preds)}  (ties count as a miss for the predicted winner)")
    print()
    show = table.copy()
    show["mean_predicted_wp"] = show["mean_predicted_wp"].map(
        lambda x: "" if pd.isna(x) else f"{100 * x:5.1f}%"
    )
    show["actual_win_rate"] = show["actual_win_rate"].map(
        lambda x: "" if pd.isna(x) else f"{100 * x:5.1f}%"
    )
    show["gap"] = show["gap"].map(lambda x: "" if pd.isna(x) else f"{100 * x:+5.1f} pp")
    print(show.to_string(index=False))


if __name__ == "__main__":
    main()
