# GAMEDAY LEVER TRACKER
Living doc — status of every model lever tested, shipped, or shelved.
Update after every experiment. Source of truth for numbers is metrics_log.csv;
this doc is the readable summary layer on top of it.

Thresholds for HELPED/HURT/NO EFFECT: ΔR² ≥ 0.005 AND ΔSU ≥ 1pp — both must clear.

---

## ✅ LIVE / SHIPPED

### Recency + turnover-luck (standing EPA layers)
- Status: LIVE, always-on (`USE_RECENCY_WEIGHT=True`, `USE_TURNOVER_ADJUSTMENT=True` in models/model_config.py)
- Isolated tests (2025 weeks 5+, 221 games, no SU/ATS on these rows):
  - v1_baseline: r=0.453, R²=0.205
  - v1_flags_off: r=0.452569, R²=0.204819
  - v1_recency: r=0.455371, R²=0.207363
  - v1_turnover_adj: r=0.449584, R²=0.202126
  - v1_recency_turnover / v2_default_recency_turnover: r=0.457704, R²=0.209493
- Standing pre-QB OOS (same 221 games; SU/ATS from later OFF baselines in the log):
  r=0.4577, R²=0.2095, SU=63.8%, ATS=51.4%
- v2_default_recency_turnover note: this pair is the standing baseline.

### QB-change layer
- Status: LIVE, always-on (`USE_QB_CHANGE=True` in models/model_config.py)
- version_tag: v3_qb_change
- Result: ΔR²=+0.0145, ΔSU=+1.4pp, ΔATS=+1.8pp — HELPED, confirmed
- OFF (recency+turnover only): r=0.4577, R²=0.2095, SU=63.8%, ATS=51.4%
- ON: r=0.4733, R²=0.2240, SU=65.2%, ATS=53.2%
  (2025 weeks 5+, walk-forward, 221 games; pick_accuracy_su=0.6515837104072398, pick_accuracy_ats=0.5318181818181819)
- Log note: isolated, not bundled. Not wired to dashboard at log time; now live in model_config.

---

## ❌ CLOSED — NO EFFECT / HURT (do not retest without new hypothesis or data)

### Pass/rush matchup splits
- Result: Hurt accuracy. Shelved.
- No version_tag in metrics_log.csv.

### 50–60% predicted-winner WP filter
- version_tags: v2_filtered_backtest, v2_ats_backtest
- v2_filtered_backtest: 125/221 games kept (dropped 96); filtered pick accuracy 73.6% vs unfiltered 63.8%; r=0.457704, R²=0.209493 (same ratings as unfiltered)
- v2_ats_backtest (closing nflverse spread_line, pushes excluded): filtered SU 73.6% (92/125), ATS 48.8% (61/125) vs 52.4% breakeven; unfiltered SU 63.8% (141/221), ATS 51.4% (113/220). pick_accuracy_su=0.736, pick_accuracy_ats=0.488
- Result: filter raises SU by dropping coin-flips; ATS is at or below breakeven. Closed.

### Edge-size / key-number filters
- Variants tested on the same 220-game 2025 pool (one exploratory pass, not independent confirmations):
  - v2_edge_segments: 0-1 pt n=38 ATS=44.7%; 1-2 pt n=51 ATS=43.1%; 2-3 pt n=38 ATS=55.3%; 3-5 pt n=47 ATS=53.2%; 5+ pt n=46 ATS=60.9% (+8.5 pp vs 52.4%). Row ATS=0.5136, games=220
  - v2_key_number_segments: near 3 or 7 (±0.5) n=120 ATS=48.3%; not near n=100 ATS=55.0%. Row ATS=0.4833, games=120
  - v2_combined_filter (edge≥3 and not near 3/7): n=46/220 ATS=56.5% (26/46). Inverse edge<3 AND near key: n=73 ATS=42.5%. Row ATS=0.5652
  - v2_combined_filter_e5 (edge≥5 and not near 3/7): n=25/220 ATS=60.0% (15/25). Row ATS=0.6
  - v2_filter_conclusion: edge≥5 alone best-supported on this pool (n=46, 60.9% ATS); key-number does not add independent signal on top of edge≥5
- Fresh seasons:
  - v2_edge5_outofsample_season2 (2024): n=50 ATS=46.0% (23/50); merged with 2025 edge≥5 (46 games, 60.9% / 28 hits): n=96 ATS=53.1% (51/96). r=0.420332, R²=0.176679, pick_accuracy_ats=0.46
  - v2_edge5_season3_pool (2023): n=50 ATS=42.0% (21/50); 3-season merged n=146 ATS=49.3% (72/146, −3.1 pp vs 52.4%). r=0.267424, R²=0.071516, pick_accuracy_ats=0.4932
- Result: Looked real on 220-game pool (60.9% ATS), collapsed to 49.3% ATS across 3 seasons merged. Multiple-comparisons trap. Closed: "no proven edge."

### Rest (binary AND continuous)
- Tested twice; hurt both times. Closed permanently unless a fundamentally different approach emerges.
- v3_rest_travel (binary rest/travel dummies, recency+turnover on, QB off): OFF r=0.4577, R²=0.2095, SU=63.8%, ATS=51.4%; ON r=0.4475, R²=0.2003, SU=62.0%, ATS=51.8% (221 games; pick_accuracy_su=0.6199, pick_accuracy_ats=0.5182). Log: rejected, hurt r and SU, no ATS benefit.
- v3_rest_continuous (days since last game, recency+turnover on, QB off): OFF r=0.4577, R²=0.2095, SU=63.8%, ATS=51.4%; ON r=0.4511, R²=0.2035, SU=61.5%, ATS=49.1% (221 games; pick_accuracy_su=0.6153846153846154, pick_accuracy_ats=0.4909090909090909). Verdict: HURT.

### Starter-change v1 — snap-count proxy (v3_starter_change)
- Method: combined OL/WR1/CB1 snap-count flag. ON TOP OF QB-change (live baseline), not vs pre-QB baseline.
- Result: OFF r=0.4733, R²=0.2240, SU=65.2%, ATS=53.2%; ON r=0.4679, R²=0.2189, SU=66.5%, ATS=51.8% (221 games; pick_accuracy_su=0.665158371040724, pick_accuracy_ats=0.5181818181818182)
- Verdict: NO EFFECT. Superseded by v2 below.

### Starter-change v2 — official injury report (v3_starter_change_v2)
- Method: nfl_data_py import_injuries(), WR1/CB1/OL flagged Out/Doubtful, QB handled separately (already live). Replaces noisy snap-count version (v3_starter_change). ON TOP OF QB-change.
- Result: OFF r=0.4733, R²=0.2240, SU=65.2%, ATS=53.2%; ON r=0.4751, R²=0.2257, SU=65.2%, ATS=51.8% (221 games; pick_accuracy_su=0.6515837104072398, pick_accuracy_ats=0.5181818181818182)
  ΔR²=+0.0018, ΔSU=+0.0pp, ΔATS=−1.4pp
- Verdict: NO EFFECT. Neither threshold cleared. Not wired to dashboard. Closed.

### Weather (v3_weather)
- Method: Open-Meteo free historical data, no paid source needed. Single combined flag: wind≥15mph OR precip>0 OR temp≤32F at kickoff; indoor/dome/closed roof = no weather effect. ON TOP OF QB-change (live baseline), not vs pre-QB baseline.
- Result: OFF r=0.4733, R²=0.2240, SU=65.2%, ATS=53.2%; ON r=0.4729, R²=0.2236, SU=64.7%, ATS=52.7% (221 games; pick_accuracy_su=0.6470588235294118, pick_accuracy_ats=0.5272727272727272)
  ΔR²=−0.0004, ΔSU=−0.5pp, ΔATS=−0.5pp
- Verdict: NO EFFECT. Neither threshold cleared. Not wired to dashboard. Closed.

---

## ⏳ TESTED, RESULT REAL BUT NOT SHIPPABLE

### Travel (v3_travel_refined)
- Method: travel + TZ shift + Denver altitude. Isolated; recency+turnover on, QB off. Not bundled.
- Result: OFF r=0.4577, R²=0.2095, SU=63.8%, ATS=51.4%; ON r=0.4504, R²=0.2029, SU=67.4%, ATS=51.4% (221 games; pick_accuracy_su=0.6742081447963801, pick_accuracy_ats=0.5136363636363637)
- Verdict: NO EFFECT (R² down, SU up; both thresholds did not clear in the same direction). Not wired to dashboard.
- Revisit: combined with market blend later, once blend is unleaked.

### Market blend (v3_blend_70 / 50 / 30)
- All three ON TOP OF QB-change (live baseline). Blend OFF is that baseline (r=0.4733, R²=0.2240, SU=65.2%, ATS=53.2%). 221 games.
- v3_blend_70 (70/30 model/close): r=0.5062, R²=0.2562, SU=67.0%, ATS=50.5% (pick_accuracy_su=0.669683257918552, pick_accuracy_ats=0.5045454545454545)
  → ΔR²=+0.0322, ΔSU=+1.81pp — HELPED on paper
- v3_blend_50 (50/50): r=0.5187, R²=0.2690, SU=65.6%, ATS=53.2% (pick_accuracy_su=0.6561085972850679, pick_accuracy_ats=0.5318181818181819)
  → ΔR²=+0.0451, ΔSU=+0.5pp — verdict=NO EFFECT
- v3_blend_30 (30/70): r=0.5228, R²=0.2733, SU=63.8%, ATS=54.5% (pick_accuracy_su=0.6380090497737556, pick_accuracy_ats=0.5454545454545454)
  → ΔR²=+0.0493, ΔSU=−1.4pp — verdict=NO EFFECT
- PROBLEM: uses nflverse spread_line = CLOSING line, not pre-kickoff.
  Leaks info a real pre-game blend wouldn't have. ATS vs that same close
  is contaminated. All three rows flagged invalid/unshippable in log.
- STATUS: BLOCKED pending real pre-kickoff odds data (paid source needed).

---

## 🔲 QUEUED / IN PROGRESS

### Odds source — real pre-kickoff data
- The Odds API: historical endpoint is PAID ONLY ($30/mo, 20K plan).
  Free tier confirmed blocked (401, all 5 spot-check games).
- OddsPapi: unverified, no account created. Docs state their historical
  coverage starts Jan 2026 — does NOT include 2025 regular season.
  Not usable for this backfill regardless of trust level.
- PLAN: pay for ONE month of The Odds API 20K plan ($30), backfill
  2023-2025 pre-kickoff spreads (~816 games × 10 credits ≈ 8,160 credits,
  fits in one month), then cancel. Going forward, use FREE live-odds
  endpoint polled weekly pre-kickoff (16 games/week ≈ 16 credits, well
  under the 500 free/month cap) to build snapshots without ongoing cost.
- STATUS: Not yet purchased. Weather lever is closed (v3_weather, NO EFFECT).
  Blend remains blocked until this backfill exists.

---

## 🛑 INFRASTRUCTURE / NON-MODEL ITEMS

### ESPN live-line scheduler (GamedayLineSnapshots)
- Status: DISABLED (no next run) as of this session.
- Reason: fed line_snapshots.csv for the movement graph / blend work,
  which is blocked on paid odds data anyway. No point accumulating data
  for a shelved feature. Revisit if/when odds backfill happens.

---

## HOW TO USE THIS DOC
- Every experiment gets ONE entry here, mirrored from metrics_log.csv.
- Move an entry between sections as its status changes (queued → tested →
  closed/shipped).
- Never re-open a ❌ CLOSED lever without a stated new hypothesis or new
  data source — this is the guardrail against re-litigating dead ends.
