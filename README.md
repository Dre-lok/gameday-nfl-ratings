# Gameday

Ratings-vs-Vegas comparison for NFL games. The model estimates team EPA (garbage-time filtered, opponent-adjusted), converts that to a predicted home margin and P(home), and puts it next to the closing or live line. It is **not** a betting product: there is no proven against-the-spread edge.

## Folder structure

```
Gameday/
  data/                     pbp CSVs, priors, calibration, snapshot + reliable-read logs
  models/                   rating pipeline, walk-forward OOS, metrics_log.csv
  odds/                     ESPN scoreboard pull + scheduled snapshot script
  gameday-dashboard/
    backend/                FastAPI (`GET /api/week/...`, reliable-reads endpoints)
    frontend/               Vite + React (Week tab + Reliable Reads Tracker)
  venv/                     Python virtualenv
```

## Run the dashboard

From two terminals (Windows). Use the project venv and Node on PATH.

**Backend** (port 8000):

```
cd C:\Users\Dre\Desktop\Vault\Gameday\gameday-dashboard\backend
C:\Users\Dre\Desktop\Vault\Gameday\venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000
```

**Frontend** (port 5173; proxies `/api` to the backend):

```
cd C:\Users\Dre\Desktop\Vault\Gameday\gameday-dashboard\frontend
$env:Path = "C:\Program Files\nodejs;" + $env:Path
npm run dev
```

Open http://localhost:5173/

Python deps: `C:\Users\Dre\Desktop\Vault\Gameday\venv\Scripts\python.exe -m pip install -r C:\Users\Dre\Desktop\Vault\Gameday\requirements.txt`

Play-by-play: `data/raw/pbp_YYYY.csv` (see `data/pull_data.py`). 2026 cold-start uses `data/priors_2026.csv` plus live ESPN lines.

## Tabs

**Week** — One season/week at a time. Table of model margin, P(home), Vegas home line, Vegas-implied win %, Gap, Confidence, Divergence. Click a row for offense/defense EPA and full kickoff time. Filter: All games vs Reliable reads only.

**Reliable Reads Tracker** — Logs this week’s Reliable-read games (`data/reliable_reads_log.csv`) and grades them Hit / Miss / Pending once a final score exists in pbp. Running record is straight-up only (did the model’s favored side win). This tracks calibration over time, not a betting record.

## Current status

- Standing model: recency + turnover-luck on; rest/travel tested and **rejected** (hurt r and SU, no ATS benefit). See `models/metrics_log.csv`.
- Walk-forward OOS is the validation standard. Filter / edge / key-number mining did **not** produce a proven ATS edge across seasons.
- Live ESPN lines overlay the **current** slate only; past weeks use stored nflverse close.
- Windows Task Scheduler job `GamedayLineSnapshots` runs `odds/run_snapshot.bat` every 2 hours (append-only `data/line_snapshots.csv`; run log `data/line_snapshot_runs.log`). The dashboard does not need to be open for snapshots.
