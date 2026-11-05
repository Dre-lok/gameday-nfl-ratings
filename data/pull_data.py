"""Pull raw NFL play-by-play data for the 2025 season."""

import io
import os
import ssl
import urllib.request
from pathlib import Path

import certifi

os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
ssl._create_default_https_context = lambda: ssl.create_default_context(
    cafile=certifi.where()
)

import nfl_data_py as nfl
import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "raw"
OUTPUT_PATH = RAW_DIR / "pbp_2025.csv"
NFLVERSE_PBP_CSV = (
    "https://github.com/nflverse/nflverse-data/releases/download/pbp/"
    "play_by_play_{year}.csv.gz"
)


def _ssl_context() -> ssl.SSLContext:
    return ssl.create_default_context(cafile=certifi.where())


def _import_pbp_csv_gz(year: int) -> pd.DataFrame:
    url = NFLVERSE_PBP_CSV.format(year=year)
    req = urllib.request.Request(url, headers={"User-Agent": "gameday-pbp-pull"})
    with urllib.request.urlopen(req, context=_ssl_context()) as resp:
        raw = resp.read()
    return pd.read_csv(io.BytesIO(raw), compression="gzip", low_memory=False)


def pull_season(year: int) -> Path:
    """Download nflverse PBP for `year` to data/raw/pbp_{year}.csv if missing."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"pbp_{year}.csv"
    if path.exists() and path.stat().st_size > 0:
        print(f"Using existing {path}")
        return path
    print(f"Pulling nflverse PBP for {year}...")
    pbp = None
    try:
        pbp = _import_pbp_csv_gz(year)
    except Exception as exc:
        print(f"csv.gz download failed ({exc}); trying nfl_data_py")
        try:
            pbp = nfl.import_pbp_data([year])
        except Exception as inner:
            print(f"nfl_data_py import failed ({inner})")
            pbp = None
    if pbp is None or len(pbp) == 0:
        raise RuntimeError(f"nflverse returned no play-by-play rows for {year}")
    pbp.to_csv(path, index=False)
    print(f"Saved {path} ({len(pbp)} rows)")
    return path


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    pbp = nfl.import_pbp_data([2025])
    pbp.to_csv(OUTPUT_PATH, index=False)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", 40)

    print("Column names:")
    print(list(pbp.columns))
    print()
    print(f"Row count: {len(pbp)}")
    print()
    print("First 5 rows:")
    print(pbp.head(5))


if __name__ == "__main__":
    main()
