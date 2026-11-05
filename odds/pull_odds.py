"""Pull current NFL spread, moneyline, and total odds from ESPN (free, no key)."""

from __future__ import annotations

import json
import os
import ssl
import urllib.request
from pathlib import Path

import certifi
import pandas as pd

os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
ssl._create_default_https_context = lambda: ssl.create_default_context(
    cafile=certifi.where()
)

ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "odds" / "raw" / "current_week.json"
SITE_URL = "https://site.web.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
SITE_API_LEGACY = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
CDN_URL = "https://cdn.espn.com/core/nfl/scoreboard?xhr=1"

# ESPN abbreviations that differ from nflverse / nfl_data_py
ESPN_TO_NFLVERSE = {
    "LAR": "LA",
    "WSH": "WAS",
    "JAC": "JAX",
}


def to_nflverse(abbr: str) -> str:
    return ESPN_TO_NFLVERSE.get(str(abbr).upper(), str(abbr).upper())


def _ssl_context() -> ssl.SSLContext:
    return ssl.create_default_context(cafile=certifi.where())


def unwrap_scoreboard(raw: dict) -> dict:
    """cdn.espn.com wraps the scoreboard in content.sbData; the site API does not."""
    content = raw.get("content") if isinstance(raw, dict) else None
    if isinstance(content, dict) and isinstance(content.get("sbData"), dict):
        return content["sbData"]
    return raw


def fetch_json(url: str) -> dict:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json,text/plain,*/*",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.espn.com/nfl/scoreboard",
        },
    )
    with urllib.request.urlopen(req, context=_ssl_context(), timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _try_urls(urls: list[str]) -> dict:
    last_err = None
    for url in urls:
        try:
            return fetch_json(url)
        except Exception as exc:  # noqa: BLE001 — try the next host
            last_err = exc
    raise RuntimeError(f"All odds endpoints failed. Last error: {last_err}") from last_err


def fetch_scoreboard() -> tuple[dict, dict, str]:
    """Return (raw_payload, scoreboard, slate_label).

    During the regular season, use ESPN's current week. In preseason / offseason,
    use the next regular-season week so we compare against real NFL games, not
    backup-heavy preseason lines.
    """
    raw = _try_urls([SITE_URL, CDN_URL, SITE_API_LEGACY])
    board = unwrap_scoreboard(raw)
    season = board.get("season") or {}
    season_type = int(season.get("type") or 0)
    week = (board.get("week") or {}).get("number")
    year = season.get("year")

    if season_type == 2 and board.get("events"):
        label = f"{year} regular season week {week}"
        return raw, board, label

    raw = _try_urls(
        [
            f"{SITE_URL}?seasontype=2&week=1",
            f"{CDN_URL}&seasontype=2&week=1&year={year or ''}",
            f"{SITE_API_LEGACY}?seasontype=2&week=1",
        ]
    )
    board = unwrap_scoreboard(raw)
    up_season = board.get("season") or {}
    up_year = up_season.get("year")
    up_week = (board.get("week") or {}).get("number")
    label = (
        f"{up_year} regular season week {up_week} "
        f"(ESPN current board is preseason/offseason type={season_type} week={week})"
    )
    return raw, board, label


def save_raw(raw: dict, board: dict, slate_label: str) -> None:
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "slate": slate_label,
        "source": "ESPN scoreboard (DraftKings odds)",
        "scoreboard": board,
    }
    RAW_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def pull_current_week() -> tuple[pd.DataFrame, str]:
    raw, board, label = fetch_scoreboard()
    save_raw(raw, board, label)
    games = parse_games(board)
    return games, label


def _parse_american(value) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(str(value).replace("+", "").strip())
    except (TypeError, ValueError):
        return None


def parse_games(raw: dict) -> pd.DataFrame:
    rows = []
    for event in raw.get("events") or []:
        comps = event.get("competitions") or []
        if not comps:
            continue
        game = comps[0]
        home = away = None
        for team in game.get("competitors") or []:
            abbr = to_nflverse((team.get("team") or {}).get("abbreviation"))
            if team.get("homeAway") == "home":
                home = abbr
            elif team.get("homeAway") == "away":
                away = abbr
        if not home or not away:
            continue

        odds_list = game.get("odds") or []
        odds = odds_list[0] if odds_list else {}
        provider = ((odds.get("provider") or {}).get("name"))

        home_spread = odds.get("spread")
        if home_spread is None:
            ps = ((odds.get("pointSpread") or {}).get("home") or {}).get("close") or {}
            home_spread = _parse_american(ps.get("line"))
        try:
            home_spread = float(home_spread) if home_spread is not None else None
        except (TypeError, ValueError):
            home_spread = None

        ml = odds.get("moneyline") or {}
        ml_home = _parse_american(((ml.get("home") or {}).get("close") or {}).get("odds"))
        ml_away = _parse_american(((ml.get("away") or {}).get("close") or {}).get("odds"))
        if ml_home is None:
            ml_home = _parse_american((odds.get("homeTeamOdds") or {}).get("moneyLine"))
        if ml_away is None:
            ml_away = _parse_american((odds.get("awayTeamOdds") or {}).get("moneyLine"))

        total = odds.get("overUnder")
        try:
            total = float(total) if total is not None else None
        except (TypeError, ValueError):
            total = None

        rows.append(
            {
                "game": f"{away} @ {home}",
                "home_team": home,
                "away_team": away,
                "kickoff": event.get("date"),
                "market_spread": home_spread,
                "market_moneyline_home": ml_home,
                "market_moneyline_away": ml_away,
                "market_total": total,
                "book": provider,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    games, label = pull_current_week()
    pd.set_option("display.width", 160)
    pd.set_option("display.max_rows", 40)
    print(f"Source: ESPN DraftKings odds  |  slate: {label}")
    print(f"Saved raw response to {RAW_PATH}")
    print()
    cols = [
        "game",
        "home_team",
        "away_team",
        "market_spread",
        "market_moneyline_home",
        "market_moneyline_away",
        "market_total",
    ]
    print(games[cols].to_string(index=False))
    print()
    print(f"{len(games)} games  |  market_spread is the home-team spread (negative = home favored)")


if __name__ == "__main__":
    main()
