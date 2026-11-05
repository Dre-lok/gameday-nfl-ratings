/** Display-only mapping of existing week-table values to Plain-view copy.

Does not change model math, confidence thresholds, or divergence rules.
*/

export type ConfidenceTier = "strong" | "lean" | "tossup";

export type PlainGame = {
  home_team: string;
  away_team: string;
  model_predicted_margin: number;
  model_win_prob_home: number;
  divergence_label: string | null;
  home_offense: number | null;
  home_defense: number | null;
  away_offense: number | null;
  away_defense: number | null;
};

function favoriteWinProb(pHome: number): { teamIsHome: boolean; p: number } {
  if (pHome >= 0.5) return { teamIsHome: true, p: pHome };
  return { teamIsHome: false, p: 1 - pHome };
}

/** Same 40/60 Confidence split, with Reliable reads split at 70% for chips. */
export function confidenceTier(pHome: number): ConfidenceTier {
  const { p } = favoriteWinProb(pHome);
  if (p >= 0.7) return "strong";
  if (p >= 0.6) return "lean";
  return "tossup";
}

export function leanPhrase(game: PlainGame): string {
  const { teamIsHome, p } = favoriteWinProb(game.model_win_prob_home);
  const team = teamIsHome ? game.home_team : game.away_team;
  if (p >= 0.7) return `Strong lean: ${team}`;
  if (p >= 0.6) return `Lean: ${team}`;
  return "Toss-up, no real read";
}

export function chipClass(tier: ConfidenceTier): string {
  if (tier === "strong") return "chip chip-green";
  if (tier === "lean") return "chip chip-amber";
  return "chip chip-gray";
}

/** Aligned = small gap; Minor/Notable = model disagrees. */
export function gapProse(divergenceLabel: string | null): string {
  if (!divergenceLabel) return "—";
  if (divergenceLabel === "Aligned") return "Model and Vegas mostly agree";
  return "Model sees this differently than Vegas";
}

function unitOf(value: number | null): number | null {
  if (value === null || Number.isNaN(value)) return null;
  return value;
}

/** Winner's offense vs opponent offense, or winner's defense (lower EPA allowed). */
export function biggestStatEdge(game: PlainGame): string | null {
  const homeWins = game.model_predicted_margin > 0 || (
    game.model_predicted_margin === 0 && game.model_win_prob_home >= 0.5
  );
  const winner = homeWins ? game.home_team : game.away_team;
  const wOff = unitOf(homeWins ? game.home_offense : game.away_offense);
  const lOff = unitOf(homeWins ? game.away_offense : game.home_offense);
  const wDef = unitOf(homeWins ? game.home_defense : game.away_defense);
  const lDef = unitOf(homeWins ? game.away_defense : game.home_defense);
  if (wOff === null || lOff === null || wDef === null || lDef === null) {
    return null;
  }
  const offenseEdge = wOff - lOff;
  const defenseEdge = lDef - wDef;
  if (defenseEdge > offenseEdge) return `${winner}'s defense`;
  return `${winner}'s offense`;
}

export function projectionSentence(game: PlainGame): string {
  const homeWins = game.model_predicted_margin > 0 || (
    game.model_predicted_margin === 0 && game.model_win_prob_home >= 0.5
  );
  const team = homeWins ? game.home_team : game.away_team;
  const by = Math.abs(game.model_predicted_margin).toFixed(1);
  const edge = biggestStatEdge(game);
  if (!edge) return `${team} projects to win by ${by}.`;
  return `${team} projects to win by ${by} - ${edge} is the biggest factor.`;
}
