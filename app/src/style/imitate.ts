// "Write like me" setting. v1 is prompt-based and private: it learns from the user's own
// messages on the device. (A per-user LoRA can replace it later without changing the setting.)

const EMOJI = /\p{Extended_Pictographic}/u;

export type StyleStats = {
  samples: number;
  avgChars: number;
  lowercaseStart: number; // share of messages starting with a lowercase letter
  emoji: number; // share of messages with an emoji
  noFinalPunct: number; // share without . ! ? at the end
};

/** Messages worth learning from: the user's own writing, not pasted text or one-word replies. */
export function usableSamples(messages: string[], max = 12): string[] {
  const out: string[] = [];
  for (let i = messages.length - 1; i >= 0 && out.length < max; i--) {
    const m = messages[i].trim();
    if (m.length < 15 || m.length > 400) continue;
    if (/```|https?:\/\//.test(m)) continue;
    if (!out.includes(m)) out.push(m);
  }
  return out.reverse();
}

export function styleStats(samples: string[]): StyleStats {
  const n = samples.length || 1;
  const share = (f: (s: string) => boolean) => Math.round((100 * samples.filter(f).length) / n) / 100;
  return {
    samples: samples.length,
    avgChars: Math.round(samples.reduce((a, s) => a + s.length, 0) / n),
    lowercaseStart: share((s) => /^\p{Ll}/u.test(s)),
    emoji: share((s) => EMOJI.test(s)),
    noFinalPunct: share((s) => !/[.!?…]$/.test(s)),
  };
}

/** Prompt block describing the user's style, or null if there is not enough to go on. */
export function buildStyleProfile(userMessages: string[], minSamples = 3): string | null {
  const samples = usableSamples(userMessages);
  if (samples.length < minSamples) return null;
  const s = styleStats(samples);
  const traits: string[] = [];
  traits.push(s.avgChars < 80 ? "short messages" : s.avgChars < 200 ? "medium-length messages" : "long messages");
  if (s.lowercaseStart >= 0.6) traits.push("usually starts sentences in lowercase");
  if (s.noFinalPunct >= 0.6) traits.push("often skips the final period");
  if (s.emoji >= 0.3) traits.push("uses emojis");
  else traits.push("rarely uses emojis");
  const examples = samples.slice(-6).map((m) => `- ${m}`).join("\n");
  return [
    "Write your answers the way this user writes: same tone, slang, length and punctuation habits.",
    "If the user asks you to write something for them (a message, an email, a post), write it as they would.",
    `Their style: ${traits.join(", ")}.`,
    "Examples of their own messages:",
    examples,
  ].join("\n");
}
