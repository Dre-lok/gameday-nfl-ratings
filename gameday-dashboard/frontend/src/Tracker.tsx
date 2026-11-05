import { useEffect, useState } from "react";

type PickRow = {
  date_logged: string;
  season: number;
  week: number;
  game_id: string;
  matchup: string;
  model_p_home: number;
  implied_side: string;
  vegas_line: number | string | null;
  status: "Pending" | "Hit" | "Miss";
  home_score: number | null;
  away_score: number | null;
};

type LogPayload = {
  picks: PickRow[];
  record: {
    hits: number;
    misses: number;
    pending: number;
    graded: number;
    win_pct: number | null;
  };
  appended?: number;
  skipped?: number;
  reliable_this_week?: number;
};

function fmtSpread(value: number | string | null): string {
  if (value === null || value === "" || Number.isNaN(Number(value))) return "—";
  const n = Number(value);
  return n > 0 ? `+${n.toFixed(1)}` : n.toFixed(1);
}

function statusClass(status: string): string {
  if (status === "Hit") return "badge badge-solid";
  return "badge badge-muted";
}

export default function ReliableReadsTracker({
  season,
  week,
}: {
  season: number;
  week: number;
}) {
  const [data, setData] = useState<LogPayload | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);

  async function refresh() {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch("/api/reliable-reads");
      if (!res.ok) {
        throw new Error("load failed");
      }
      setData(await res.json());
    } catch {
      setData(null);
      setError("Couldn't load data, try again");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void refresh();
  }, []);

  async function logWeek() {
    setBusy(true);
    setError(null);
    setNotice(null);
    try {
      const res = await fetch(`/api/reliable-reads/log/${season}/${week}`, {
        method: "POST",
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail || `HTTP ${res.status}`);
      }
      const payload: LogPayload = await res.json();
      setData(payload);
      const appended = payload.appended ?? 0;
      const skipped = payload.skipped ?? 0;
      setNotice(
        `Logged ${appended} new game${appended === 1 ? "" : "s"} from ${season} week ${week}` +
          (skipped ? ` (${skipped} already logged)` : "") +
          (payload.reliable_this_week === 0 ? " — no Reliable reads this week" : ""),
      );
    } catch {
      setError("Couldn't load data, try again");
    } finally {
      setBusy(false);
    }
  }

  const rec = data?.record;
  const recordLine = rec
    ? rec.graded > 0
      ? `${rec.hits}-${rec.misses}, ${((rec.win_pct ?? 0) * 100).toFixed(0)}%`
      : `${rec.hits}-${rec.misses}`
    : "0-0";

  return (
    <>
      <p className="sub">
        Tracks the model's own calibration over time — grading whether
        Reliable-read games actually hit at the rate the model claims. Not a
        betting record.
      </p>
      {!loading && rec ? (
        <p className="status">
          Record: {recordLine}
          {rec.pending ? ` · ${rec.pending} pending` : ""}
          {rec.graded ? ` · ${rec.graded} graded` : ""}
        </p>
      ) : null}
      <div className="controls">
        <button type="button" disabled={busy} onClick={() => void logWeek()}>
          {busy ? "Logging…" : "Log this week's Reliable Reads"}
        </button>
      </div>
      <p className="caption">
        Uses season {season} week {week} from the controls above. Straight-up
        only: did the side the model favored actually win.
      </p>
      {loading ? <p className="status">Loading...</p> : null}
      {notice ? <p className="status">{notice}</p> : null}
      {error ? <p className="status error">{error}</p> : null}
      {!loading && data && data.picks.length === 0 ? (
        <p className="status">No Reliable reads logged yet.</p>
      ) : null}
      {!loading && data && data.picks.length > 0 ? (
        <table>
          <thead>
            <tr>
              <th>Date logged</th>
              <th>Week</th>
              <th>Matchup</th>
              <th>Implied side</th>
              <th className="num">Model P(home)</th>
              <th className="num">Vegas (home)</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
            {data.picks.map((pick) => (
              <tr key={pick.game_id}>
                <td>{String(pick.date_logged).slice(0, 10)}</td>
                <td>
                  {pick.season} / {pick.week}
                </td>
                <td>{pick.matchup}</td>
                <td>{pick.implied_side}</td>
                <td className="num">{(100 * Number(pick.model_p_home)).toFixed(1)}%</td>
                <td className="num">{fmtSpread(pick.vegas_line)}</td>
                <td>
                  <span className={statusClass(pick.status)}>{pick.status}</span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : null}
    </>
  );
}
