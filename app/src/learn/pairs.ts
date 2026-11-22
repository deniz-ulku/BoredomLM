// On-device learning, data side: turn what the user does into preference pairs, the same
// format the server pipeline trains on (prompt messages + chosen + rejected). Everything stays
// on the phone. The nightly trainer (native, see docs/ON_DEVICE_LEARNING.md) consumes them.

export type PairSource = "edit" | "regenerate";

export type PromptTurn = { role: "user" | "assistant"; content: string };

export type PreferencePair = {
  id: string;
  createdAt: number;
  source: PairSource;
  /** Conversation up to and including the user turn being answered (no system prompt, no images). */
  prompt: PromptTurn[];
  chosen: string;
  rejected: string;
  /** Set once a nightly run has trained on it. */
  trainedAt?: number;
};

export const MAX_PAIRS = 2000;
const CONTEXT_TURNS = 6;

/** Words that differ between two texts, as a share of the longer one (0 = identical). */
export function changeRatio(a: string, b: string): number {
  const wa = a.trim().split(/\s+/).filter(Boolean);
  const wb = b.trim().split(/\s+/).filter(Boolean);
  const n = Math.max(wa.length, wb.length);
  if (n === 0) return 0;
  // LCS on words, bounded for long texts.
  const A = wa.slice(0, 600), B = wb.slice(0, 600);
  let prev = new Array<number>(B.length + 1).fill(0);
  for (let i = 1; i <= A.length; i++) {
    const cur = new Array<number>(B.length + 1).fill(0);
    for (let j = 1; j <= B.length; j++) cur[j] = A[i - 1] === B[j - 1] ? prev[j - 1] + 1 : Math.max(prev[j], cur[j - 1]);
    prev = cur;
  }
  return 1 - prev[B.length] / Math.max(A.length, B.length);
}

/** Why a pair is not worth learning from, or null if it is. */
export function rejectReason(chosen: string, rejected: string): string | null {
  const norm = (t: string) => t.trim().replace(/\s+/g, " ");
  if (!norm(chosen) || !norm(rejected)) return "empty";
  // Only whitespace differs: nothing to learn. A changed "!" or comma is a real style signal.
  if (norm(chosen) === norm(rejected)) return "identical";
  return null;
}

export function contextFor(turns: PromptTurn[]): PromptTurn[] {
  const t = turns.slice(-CONTEXT_TURNS);
  while (t.length && t[0].role !== "user") t.shift();
  return t;
}

export function makePair(
  source: PairSource, prompt: PromptTurn[], chosen: string, rejected: string, now: number, id: string,
): PreferencePair | null {
  if (rejectReason(chosen, rejected)) return null;
  const ctx = contextFor(prompt);
  if (!ctx.length || ctx[ctx.length - 1].role !== "user") return null;
  return { id, createdAt: now, source, prompt: ctx, chosen: chosen.trim(), rejected: rejected.trim() };
}

/** Add a pair, replacing an older one for the same prompt+rejected, and keep the newest MAX_PAIRS. */
export function addPair(pairs: PreferencePair[], p: PreferencePair, max = MAX_PAIRS): PreferencePair[] {
  const key = (x: PreferencePair) => JSON.stringify([x.prompt[x.prompt.length - 1]?.content, x.rejected]);
  const k = key(p);
  const out = pairs.filter((x) => key(x) !== k);
  out.push(p);
  out.sort((a, b) => a.createdAt - b.createdAt);
  return out.slice(-max);
}

/** Pairs a nightly run should use: untrained first, then a replay share of older ones. */
export function nightlyBatch(pairs: PreferencePair[], maxNew = 64, replay = 0.25): PreferencePair[] {
  const fresh = pairs.filter((p) => !p.trainedAt).slice(-maxNew);
  const old = pairs.filter((p) => p.trainedAt);
  const nReplay = Math.min(old.length, Math.ceil(fresh.length * replay));
  return [...old.slice(-nReplay), ...fresh];
}

/** Regenerations: when the user moves on, the version they kept beats the ones they replaced. */
export function regenerationPairs(
  prompt: PromptTurn[], versions: string[], kept: string, now: number, newId: () => string,
): PreferencePair[] {
  return versions
    .filter((v) => v !== kept)
    .map((v) => makePair("regenerate", prompt, kept, v, now, newId()))
    .filter((p): p is PreferencePair => p !== null);
}
