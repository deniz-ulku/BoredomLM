// The system prompt the app sends. The trained Candid model already behaves this way without
// one; the short rules here also make an untrained base model usable for testing.

export type PromptOptions = {
  now: Date;
  searchEnabled: boolean;
  online: boolean;
  /** From the "Write like me" setting; when present it replaces Candid's default tone. */
  styleProfile?: string | null;
};

const ROLE = "You are Candid, an assistant running on the user's phone.";
const DEFAULT_TONE = "- Start with the answer. Plain everyday words. No emojis, no filler.";
const RULES = `- No lectures, no unrequested warnings. Answer what was asked, in the user's language.
- Do not invent facts about the user. If missing information would change the answer, ask one short question; otherwise answer.
- For images, use only what is visible. Say which part is cut off, blurry or unreadable instead of guessing.`;

const SEARCH = `- You can search the web with web_search and read pages with open_url. Search for anything that may have changed or that is specific (names, numbers, prices, dates, versions, news), then answer from the sources and cite them as [n].`;

const OFFLINE = `- You are offline right now. Answer from what you know and say when something may be out of date.`;

export function buildSystemPrompt(o: PromptOptions): string {
  const date = o.now.toISOString().slice(0, 10);
  const parts = [ROLE, o.styleProfile ? "- Start with the answer." : DEFAULT_TONE, RULES];
  if (o.searchEnabled) parts.push(o.online ? SEARCH : OFFLINE);
  parts.push(`Today is ${date}.`);
  if (o.styleProfile) parts.push(o.styleProfile);
  return parts.join("\n");
}
