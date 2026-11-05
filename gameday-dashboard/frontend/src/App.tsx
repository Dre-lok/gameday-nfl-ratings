import { FormEvent, Fragment, useEffect, useState } from "react";
import ReliableReadsTracker from "./Tracker";
import {
  chipClass,
  confidenceTier,
  gapProse,
  leanPhrase,
  projectionSentence,
} from "./plainView";

type DisplayMode = "stat" | "plain";

type Game = {
  game_id: string;
  game_date: string | null;
  game_datetime: string | null;
  home_team: string;
  away_team: string;
  model_predicted_margin: number;
  model_win_prob_home: number;
  vegas_spread: number | null;
  vegas_implied_prob: number | null;
  divergence_label: string | null;
  vegas_source?: string | null;
  home_offense: number | null;
  home_defense: number | null;
  away_offense: number | null;
  away_defense: number | null;
};

type WeekPayload = {
  season: number;
  week: number;
  games: Game[];
};

function fmtSigned(value: number): string {
  return value > 0 ? `+${value.toFixed(1)}` : value.toFixed(1);
}

function fmtSpread(value: number | null): string {
  if (value === null || Number.isNaN(value)) return "—";
  return fmtSigned(value);
}

function fmtPct(value: number | null): string {
  if (value === null || Number.isNaN(value)) return "—";
  return `${(100 * value).toFixed(1)}%`;
}

function divergenceBadgeClass(label: string | null): string {
  if (label === "Notable divergence") return "badge badge-solid";
  return "badge badge-muted";
}

function fmtRating(value: number | null): string {
  if (value === null || Number.isNaN(value)) return "—";
  return value.toFixed(3);
}

function fmtDate(iso: string | null): string {
  if (!iso) return "—";
  const d = new Date(`${iso}T12:00:00`);
  if (Number.isNaN(d.getTime())) return "—";
  const days = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
  return `${days[d.getDay()]} ${d.getMonth() + 1}/${d.getDate()}`;
}

function fmtDateTime(value: string | null): string {
  if (!value) return "—";
  const parsed = value.includes("T")
    ? new Date(value)
    : value.length === 10
      ? new Date(`${value}T12:00:00`)
      : new Date(value);
  if (Number.isNaN(parsed.getTime())) return value;
  return parsed.toLocaleString(undefined, {
    weekday: "long",
    year: "numeric",
    month: "long",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
    timeZoneName: "short",
  });
}

function modelVegasGap(modelMargin: number, vegasHome: number | null): string {
  if (vegasHome === null || Number.isNaN(vegasHome)) return "—";
  // Vegas home line is negative when home is favored. Implied home margin is -line.
  const impliedVegasMargin = -vegasHome;
  return fmtSigned(modelMargin - impliedVegasMargin);
}

function confidenceLabel(pHome: number): "Reliable read" | "Toss-up zone" {
  if (pHome >= 0.6 || pHome <= 0.4) return "Reliable read";
  return "Toss-up zone";
}

function displayedGames(games: Game[], reliableOnly: boolean): Game[] {
  if (!reliableOnly) return games;
  return games
    .filter((game) => confidenceLabel(game.model_win_prob_home) === "Reliable read")
    .slice()
    .sort((a, b) => {
      const da = Math.abs(a.model_win_prob_home - 0.5);
      const db = Math.abs(b.model_win_prob_home - 0.5);
      return db - da;
    });
}

function GameBreakdown({ game }: { game: Game }) {
  const conf = confidenceLabel(game.model_win_prob_home);
  return (
    <div className="breakdown">
      <p className="breakdown-when">{fmtDateTime(game.game_datetime || game.game_date)}</p>
      <div className="breakdown-hero">
        <div>
          <span className="k">Model margin</span>
          <span className="v">{fmtSigned(game.model_predicted_margin)}</span>
        </div>
        <div>
          <span className="k">P(home)</span>
          <span className="v">{(100 * game.model_win_prob_home).toFixed(1)}%</span>
        </div>
      </div>
      <div className="breakdown-meta">
        <div>
          <span className="k">Vegas (home)</span>
          <span>{fmtSpread(game.vegas_spread)}</span>
        </div>
        <div>
          <span className="k">Vegas Implied %</span>
          <span>{fmtPct(game.vegas_implied_prob)}</span>
        </div>
        <div>
          <span className="k">Gap</span>
          <span>{modelVegasGap(game.model_predicted_margin, game.vegas_spread)}</span>
        </div>
        <div>
          <span className="k">Confidence</span>
          <span
            className={
              conf === "Reliable read" ? "badge badge-solid" : "badge badge-muted"
            }
          >
            {conf}
          </span>
        </div>
        <div>
          <span className="k">Divergence</span>
          {game.divergence_label ? (
            <span className={divergenceBadgeClass(game.divergence_label)}>
              {game.divergence_label}
            </span>
          ) : (
            "—"
          )}
        </div>
      </div>
      <table className="ratings">
        <thead>
          <tr>
            <th>Ratings (EPA/play)</th>
            <th className="num">Offense</th>
            <th className="num">Defense</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>
              Home {game.home_team}
            </td>
            <td className="num">{fmtRating(game.home_offense)}</td>
            <td className="num">{fmtRating(game.home_defense)}</td>
          </tr>
          <tr>
            <td>
              Away {game.away_team}
            </td>
            <td className="num">{fmtRating(game.away_offense)}</td>
            <td className="num">{fmtRating(game.away_defense)}</td>
          </tr>
        </tbody>
      </table>
    </div>
  );
}

export default function App() {
  const [season, setSeason] = useState(2025);
  const [week, setWeek] = useState(10);
  const [data, setData] = useState<WeekPayload | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [reliableOnly, setReliableOnly] = useState(false);
  const [displayMode, setDisplayMode] = useState<DisplayMode>("stat");
  const [tab, setTab] = useState<"week" | "tracker">("week");

  async function loadWeek(nextSeason: number, nextWeek: number) {
    setLoading(true);
    setError(null);
    setExpandedId(null);
    setReliableOnly(false);
    try {
      const res = await fetch(`/api/week/${nextSeason}/${nextWeek}`);
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail || `HTTP ${res.status}`);
      }
      const payload: WeekPayload = await res.json();
      setData(payload);
    } catch (err) {
      setData(null);
      setError(err instanceof Error ? err.message : "request failed");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void loadWeek(2025, 10);
  }, []);

  function onSubmit(event: FormEvent) {
    event.preventDefault();
    void loadWeek(season, week);
  }

  function toggleRow(gameId: string) {
    setExpandedId((current) => (current === gameId ? null : gameId));
  }

  const rows = data ? displayedGames(data.games, reliableOnly) : [];

  return (
    <>
      <h1>Gameday</h1>
      <div className="tabs">
        <button
          type="button"
          className={tab === "week" ? "tab-on" : ""}
          onClick={() => setTab("week")}
        >
          Week
        </button>
        <button
          type="button"
          className={tab === "tracker" ? "tab-on" : ""}
          onClick={() => setTab("tracker")}
        >
          Reliable Reads Tracker
        </button>
      </div>
      {tab === "week" ? (
        <p className="sub">Model vs Vegas comparison.</p>
      ) : null}
      <form className="controls" onSubmit={onSubmit}>
        <label>
          Season
          <input
            type="number"
            value={season}
            onChange={(e) => setSeason(Number(e.target.value))}
            min={2023}
            max={2026}
          />
        </label>
        <label>
          Week
          <input
            type="number"
            value={week}
            onChange={(e) => setWeek(Number(e.target.value))}
            min={1}
            max={22}
          />
        </label>
        <button type="submit" disabled={loading}>
          {loading ? "Loading…" : "Load"}
        </button>
      </form>
      {tab === "tracker" ? (
        <ReliableReadsTracker season={season} week={week} />
      ) : (
        <>
      {loading ? <p className="status">Loading...</p> : null}
      {!loading && error ? (
        <p className="status error">Couldn't load data, try again</p>
      ) : null}
      {!loading && data && !error ? (
        <p className="status">
          {data.season} week {data.week} —{" "}
          {reliableOnly
            ? `${rows.length} of ${data.games.length} games (reliable reads)`
            : `${data.games.length} games`}
        </p>
      ) : null}
      {!loading && data && data.games.length > 0 ? (
        <>
          <div className="filter-bar">
            <span>Show:</span>
            <label>
              <input
                type="radio"
                name="game-filter"
                checked={!reliableOnly}
                onChange={() => {
                  setReliableOnly(false);
                  setExpandedId(null);
                }}
              />
              All games
            </label>
            <label>
              <input
                type="radio"
                name="game-filter"
                checked={reliableOnly}
                onChange={() => {
                  setReliableOnly(true);
                  setExpandedId(null);
                }}
              />
              Reliable reads only
            </label>
          </div>
          <div className="filter-bar">
            <span>Display:</span>
            <label>
              <input
                type="radio"
                name="display-mode"
                checked={displayMode === "stat"}
                onChange={() => setDisplayMode("stat")}
              />
              Stat view
            </label>
            <label>
              <input
                type="radio"
                name="display-mode"
                checked={displayMode === "plain"}
                onChange={() => setDisplayMode("plain")}
              />
              Plain view
            </label>
          </div>
          {rows.length > 0 ? (
          <table>
            <thead>
              {displayMode === "stat" ? (
                <tr>
                  <th>Matchup</th>
                  <th>Date</th>
                  <th className="num">Model margin</th>
                  <th className="num">Model P(home)</th>
                  <th className="num">Vegas (home)</th>
                  <th className="num">Vegas Implied %</th>
                  <th className="num">Model–Vegas Gap</th>
                  <th>Confidence</th>
                  <th>Divergence</th>
                </tr>
              ) : (
                <tr>
                  <th>Matchup</th>
                  <th>Date</th>
                  <th>Read</th>
                  <th>Model vs Vegas</th>
                  <th>Projection</th>
                </tr>
              )}
            </thead>
            <tbody>
              {rows.map((game) => {
                const conf = confidenceLabel(game.model_win_prob_home);
                const open = expandedId === game.game_id;
                const tier = confidenceTier(game.model_win_prob_home);
                return (
                  <Fragment key={game.game_id}>
                    <tr
                      className={open ? "row-click row-open" : "row-click"}
                      onClick={() => toggleRow(game.game_id)}
                    >
                      {displayMode === "stat" ? (
                        <>
                          <td>
                            {game.away_team} @ {game.home_team}
                          </td>
                          <td>{fmtDate(game.game_date)}</td>
                          <td className="num">{fmtSigned(game.model_predicted_margin)}</td>
                          <td className="num">{(100 * game.model_win_prob_home).toFixed(1)}%</td>
                          <td className="num">{fmtSpread(game.vegas_spread)}</td>
                          <td className="num">{fmtPct(game.vegas_implied_prob)}</td>
                          <td className="num">
                            {modelVegasGap(game.model_predicted_margin, game.vegas_spread)}
                          </td>
                          <td>
                            <span
                              className={
                                conf === "Reliable read" ? "badge badge-solid" : "badge badge-muted"
                              }
                            >
                              {conf}
                            </span>
                          </td>
                          <td>
                            {game.divergence_label ? (
                              <span className={divergenceBadgeClass(game.divergence_label)}>
                                {game.divergence_label}
                              </span>
                            ) : (
                              "—"
                            )}
                          </td>
                        </>
                      ) : (
                        <>
                          <td>
                            {game.away_team} @ {game.home_team}
                          </td>
                          <td>{fmtDate(game.game_date)}</td>
                          <td>
                            <span className={chipClass(tier)}>{leanPhrase(game)}</span>
                          </td>
                          <td>{gapProse(game.divergence_label)}</td>
                          <td className="plain-proj">{projectionSentence(game)}</td>
                        </>
                      )}
                    </tr>
                    {open ? (
                      <tr className="detail-row">
                        <td colSpan={displayMode === "stat" ? 9 : 5}>
                          <GameBreakdown game={game} />
                        </td>
                      </tr>
                    ) : null}
                  </Fragment>
                );
              })}
            </tbody>
          </table>
          ) : (
            <p className="status">No reliable reads this week.</p>
          )}
          <p className="caption">
            {displayMode === "stat"
              ? "Gap shown for transparency only - not a proven predictive signal (see model notes). Divergence flags where the model and market probabilities differ - not a proven predictive signal, see model notes."
              : "Plain view rewrites the same numbers in everyday language. It does not change the model. Reads are not a suggestion to take a side."}
          </p>
          <p className="caption">
            Confidence reflects how reliable the model's win probability has been
            historically at this range - not a prediction of this game's outcome.
          </p>
        </>
      ) : null}
        </>
      )}
    </>
  );
}
