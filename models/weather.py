"""Game-level weather lever from Open-Meteo historical hourly data.

Indoor/dome/closed-roof games are flagged as no weather effect and are not
queried. Outdoor games use temperature, wind, and precipitation at kickoff
at the stadium location. International sites that nflverse mislabels are
overridden to the actual venue.
"""

from __future__ import annotations

import json
import os
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from rating_v1 import ROOT
from rest_travel import STADIUM_COORDS

try:
    import certifi

    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
    ssl._create_default_https_context = lambda: ssl.create_default_context(
        cafile=certifi.where()
    )
except ImportError:
    certifi = None  # type: ignore[assignment]

WEATHER_PATH = ROOT / "data" / "raw" / "weather_2023_2025.csv"
SCHEDULE_PATH = ROOT / "data" / "raw" / "schedules_2023_2025.csv"
SEASONS = (2023, 2024, 2025)
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
INDOOR_ROOFS = {"dome", "closed"}
WIND_MPH_MIN = 15.0
TEMP_F_MAX = 32.0
PRECIP_MM_MIN = 0.0  # any precipitation counts as present

# Known international venues keyed by nflverse stadium_id.
STADIUM_ID_COORDS: dict[str, tuple[float, float]] = {
    "LON00": (51.5560, -0.2796),  # Wembley
    "LON02": (51.6043, -0.0664),  # Tottenham
    "FRA00": (50.0686, 8.6455),  # Deutsche Bank Park
    "GER00": (48.2188, 11.6247),  # Allianz Arena
    "SAO00": (-23.5453, -46.4730),  # Arena Corinthians
}

# 2025 international games that nflverse still tags with the US home stadium.
GAME_VENUE_OVERRIDE: dict[str, dict] = {
    "2025_01_KC_LAC": {
        "lat": -23.5453,
        "lon": -46.4730,
        "stadium": "Arena Corinthians",
        "indoor": False,
    },
    "2025_04_MIN_PIT": {
        "lat": 53.3607,
        "lon": -6.2512,
        "stadium": "Croke Park",
        "indoor": False,
    },
    "2025_05_MIN_CLE": {
        "lat": 51.6043,
        "lon": -0.0664,
        "stadium": "Tottenham Hotspur Stadium",
        "indoor": False,
    },
    "2025_06_DEN_NYJ": {
        "lat": 51.6043,
        "lon": -0.0664,
        "stadium": "Tottenham Hotspur Stadium",
        "indoor": False,
    },
    "2025_07_LA_JAX": {
        "lat": 51.5560,
        "lon": -0.2796,
        "stadium": "Wembley Stadium",
        "indoor": False,
    },
    "2025_10_ATL_IND": {
        "lat": 52.5147,
        "lon": 13.2394,
        "stadium": "Olympiastadion Berlin",
        "indoor": False,
    },
    "2025_11_WAS_MIA": {
        "lat": 40.4530,
        "lon": -3.6883,
        "stadium": "Santiago Bernabeu",
        "indoor": False,
    },
}


def _ssl_context() -> ssl.SSLContext:
    if certifi is not None:
        return ssl.create_default_context(cafile=certifi.where())
    return ssl.create_default_context()


def load_schedules(years: tuple[int, ...] = SEASONS) -> pd.DataFrame:
    years = tuple(years)
    if SCHEDULE_PATH.exists() and SCHEDULE_PATH.stat().st_size > 0:
        cached = pd.read_csv(SCHEDULE_PATH)
        have = set(pd.to_numeric(cached["season"], errors="coerce").dropna().astype(int))
        if set(years).issubset(have):
            return cached.loc[cached["season"].isin(years)].copy()
    import nfl_data_py as nfl

    sched = nfl.import_schedules(list(years))
    SCHEDULE_PATH.parent.mkdir(parents=True, exist_ok=True)
    sched.to_csv(SCHEDULE_PATH, index=False)
    return sched


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> datetime:
    """Naive local date: n-th weekday (Mon=0) of month."""
    d = datetime(year, month, 1)
    shift = (weekday - d.weekday()) % 7
    return d + timedelta(days=shift + 7 * (n - 1))


def _et_to_utc(local: datetime) -> datetime:
    """Convert naive Eastern wall time to UTC (US DST rules, no tzdata needed)."""
    dst_start = _nth_weekday(local.year, 3, 6, 2).replace(hour=2)
    dst_end = _nth_weekday(local.year, 11, 6, 1).replace(hour=2)
    hours = 4 if dst_start <= local < dst_end else 5
    return local.replace(tzinfo=timezone.utc) + timedelta(hours=hours)


def _kickoff_utc(gameday, gametime) -> datetime | None:
    if pd.isna(gameday):
        return None
    day = pd.to_datetime(gameday).strftime("%Y-%m-%d")
    clock = "13:00"
    if pd.notna(gametime) and str(gametime).strip():
        clock = str(gametime).strip()
        if len(clock) == 5:
            clock = clock + ":00"
    try:
        local = datetime.strptime(f"{day} {clock}", "%Y-%m-%d %H:%M:%S")
    except ValueError:
        try:
            local = datetime.strptime(f"{day} {clock}", "%Y-%m-%d %H:%M")
        except ValueError:
            return None
    return _et_to_utc(local)


def _venue_for_row(rec) -> tuple[float | None, float | None, bool, str]:
    """Return (lat, lon, indoor, label)."""
    gid = str(rec.game_id)
    if gid in GAME_VENUE_OVERRIDE:
        ov = GAME_VENUE_OVERRIDE[gid]
        return ov["lat"], ov["lon"], bool(ov["indoor"]), ov["stadium"]
    roof = str(rec.roof).strip().lower() if pd.notna(rec.roof) else ""
    indoor = roof in INDOOR_ROOFS
    sid = str(rec.stadium_id) if pd.notna(rec.stadium_id) else ""
    if sid in STADIUM_ID_COORDS:
        lat, lon = STADIUM_ID_COORDS[sid]
        return lat, lon, indoor, str(rec.stadium) if pd.notna(rec.stadium) else sid
    home = str(rec.home_team)
    coords = STADIUM_COORDS.get(home)
    if coords is None:
        return None, None, indoor, str(rec.stadium) if pd.notna(rec.stadium) else home
    return coords[0], coords[1], indoor, str(rec.stadium) if pd.notna(rec.stadium) else home


def _fetch_hourly(lat: float, lon: float, start: str, end: str) -> dict:
    params = {
        "latitude": f"{lat:.4f}",
        "longitude": f"{lon:.4f}",
        "start_date": start,
        "end_date": end,
        "hourly": "temperature_2m,wind_speed_10m,precipitation",
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "timezone": "UTC",
    }
    url = ARCHIVE_URL + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "gameday-weather-pull"})
    last_err: Exception | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, context=_ssl_context(), timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_err = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Open-Meteo fetch failed for {lat},{lon} {start}..{end}: {last_err}")


def _nearest_hour(hourly: dict, kickoff: datetime) -> tuple[str | None, float | None, float | None, float | None]:
    times = hourly.get("time") or []
    if not times:
        return None, None, None, None
    temps = hourly.get("temperature_2m") or [None] * len(times)
    winds = hourly.get("wind_speed_10m") or [None] * len(times)
    precips = hourly.get("precipitation") or [None] * len(times)
    best_i = 0
    best_dt = abs(
        datetime.fromisoformat(times[0]).replace(tzinfo=timezone.utc) - kickoff
    )
    for i, stamp in enumerate(times):
        ts = datetime.fromisoformat(stamp).replace(tzinfo=timezone.utc)
        delta = abs(ts - kickoff)
        if delta < best_dt:
            best_dt = delta
            best_i = i
    if best_dt > timedelta(hours=2):
        return None, None, None, None
    return times[best_i], temps[best_i], winds[best_i], precips[best_i]


def _adverse(temp_f, wind_mph, precip_mm) -> tuple[int, int, int, int]:
    wind_flag = int(pd.notna(wind_mph) and float(wind_mph) >= WIND_MPH_MIN)
    precip_flag = int(pd.notna(precip_mm) and float(precip_mm) > PRECIP_MM_MIN)
    cold_flag = int(pd.notna(temp_f) and float(temp_f) <= TEMP_F_MAX)
    return int(bool(wind_flag or precip_flag or cold_flag)), wind_flag, precip_flag, cold_flag


def pull_weather(years: tuple[int, ...] = SEASONS, sched: pd.DataFrame | None = None) -> pd.DataFrame:
    """Pull Open-Meteo kickoff weather for outdoor games in `years`; cache to disk."""
    if sched is None:
        sched = load_schedules(years)
    sched = sched.loc[sched["season"].isin(years)].copy()
    if WEATHER_PATH.exists() and WEATHER_PATH.stat().st_size > 0:
        cached = pd.read_csv(WEATHER_PATH)
        have = set(cached["game_id"].astype(str))
        need = set(sched["game_id"].astype(str))
        if need.issubset(have):
            return cached.loc[cached["game_id"].astype(str).isin(need)].copy()

    rows = []
    outdoor = []
    for rec in sched.itertuples(index=False):
        lat, lon, indoor, label = _venue_for_row(rec)
        kick = _kickoff_utc(rec.gameday, rec.gametime)
        base = {
            "game_id": rec.game_id,
            "season": int(rec.season),
            "week": int(rec.week),
            "home_team": rec.home_team,
            "away_team": rec.away_team,
            "stadium_id": rec.stadium_id,
            "stadium": label,
            "roof": rec.roof,
            "indoor": int(indoor),
            "gameday": rec.gameday,
            "gametime": rec.gametime,
            "kickoff_utc": kick.isoformat() if kick else None,
            "lat": lat,
            "lon": lon,
            "temp_f": None,
            "wind_mph": None,
            "precip_mm": None,
            "hour_utc": None,
            "weather_adverse": 0,
            "wind_flag": 0,
            "precip_flag": 0,
            "cold_flag": 0,
            "source": "indoor_skip" if indoor else "open-meteo",
        }
        if indoor or lat is None or lon is None or kick is None:
            if not indoor:
                base["source"] = "missing_venue_or_kickoff"
            rows.append(base)
        else:
            outdoor.append((base, lat, lon, kick))

    by_loc: dict[tuple[float, float], list] = {}
    for item in outdoor:
        key = (round(item[1], 3), round(item[2], 3))
        by_loc.setdefault(key, []).append(item)

    for (lat_k, lon_k), items in by_loc.items():
        by_year: dict[int, list] = {}
        for item in items:
            by_year.setdefault(item[3].year, []).append(item)
        for year, year_items in sorted(by_year.items()):
            dates = [it[3].date() for it in year_items]
            start = min(dates).isoformat()
            end = max(dates).isoformat()
            body = _fetch_hourly(year_items[0][1], year_items[0][2], start, end)
            hourly = body.get("hourly") or {}
            time.sleep(0.2)
            for base, _, _, kick in year_items:
                hour, temp_f, wind_mph, precip_mm = _nearest_hour(hourly, kick)
                base["hour_utc"] = hour
                base["temp_f"] = temp_f
                base["wind_mph"] = wind_mph
                base["precip_mm"] = precip_mm
                adv, wf, pf, cf = _adverse(temp_f, wind_mph, precip_mm)
                base["weather_adverse"] = adv
                base["wind_flag"] = wf
                base["precip_flag"] = pf
                base["cold_flag"] = cf
                if hour is None:
                    base["source"] = "open-meteo_no_hour"
                rows.append(base)

    out = pd.DataFrame(rows).sort_values(["season", "week", "game_id"])
    WEATHER_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(WEATHER_PATH, index=False)
    return out


def weather_features(pbp: pd.DataFrame, weather: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per pbp game: weather_adverse plus diagnostic columns."""
    if weather is None:
        weather = pull_weather()
    games = (
        pbp.dropna(subset=["game_id"])
        .drop_duplicates("game_id")[["game_id"]]
        .copy()
    )
    keep = [
        "game_id",
        "indoor",
        "temp_f",
        "wind_mph",
        "precip_mm",
        "weather_adverse",
        "wind_flag",
        "precip_flag",
        "cold_flag",
        "source",
        "stadium",
        "roof",
    ]
    feat = games.merge(weather[keep], on="game_id", how="left")
    for col in ("weather_adverse", "wind_flag", "precip_flag", "cold_flag", "indoor"):
        feat[col] = feat[col].fillna(0).astype(int)
    return feat
