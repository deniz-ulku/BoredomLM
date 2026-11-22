import { describe, expect, it } from "vitest";

import {
  addPair, changeRatio, contextFor, makePair, nightlyBatch, regenerationPairs, rejectReason, type PreferencePair,
} from "../src/learn/pairs";

const prompt = [
  { role: "user" as const, content: "hi" },
  { role: "assistant" as const, content: "hello" },
  { role: "user" as const, content: "write a short reply to my boss saying I'll be late" },
];

describe("learning pairs", () => {
  it("measures how much an edit changed", () => {
    expect(changeRatio("a b c d", "a b c d")).toBe(0);
    expect(changeRatio("a b c d", "a b x d")).toBe(0.25);
    expect(changeRatio("", "")).toBe(0);
  });

  it("skips pairs with nothing to learn", () => {
    expect(rejectReason("same", "same")).toBe("identical");
    expect(rejectReason("", "x")).toBe("empty");
    expect(rejectReason("Hello  there\n friend.", "Hello there friend.")).toBe("identical");
    expect(rejectReason("Hello there friend.", "Hello there friend!")).toBeNull(); // style signal
    expect(rejectReason("hey, running 15 min late", "Dear Sir, I regret to inform you...")).toBeNull();
  });

  it("builds an edit pair with bounded context that ends on the user turn", () => {
    const p = makePair("edit", prompt, "hey, 15 min late, sorry", "Dear Sir, I regret...", 1, "a")!;
    expect(p.prompt.at(-1)).toMatchObject({ role: "user" });
    expect(p).toMatchObject({ source: "edit", chosen: "hey, 15 min late, sorry" });
    expect(makePair("edit", prompt.slice(0, 2), "x y", "z w", 1, "b")).toBeNull(); // last turn not user
    const long = Array.from({ length: 10 }, (_, i) => ({ role: i % 2 ? "assistant" as const : "user" as const,
      content: `t${i}` }));
    expect(contextFor(long)[0].role).toBe("user");
    expect(contextFor(long).length).toBeLessThanOrEqual(6);
  });

  it("dedupes and caps the store", () => {
    const mk = (n: number, rejected = "r" + n): PreferencePair =>
      makePair("edit", prompt, "chosen text " + n, rejected, n, "id" + n)!;
    let all: PreferencePair[] = [];
    for (let i = 0; i < 5; i++) all = addPair(all, mk(i), 3);
    expect(all.map((p) => p.id)).toEqual(["id2", "id3", "id4"]);
    all = addPair(all, mk(9, "r4"), 3); // same prompt + rejected: replaces
    expect(all.map((p) => p.id)).toEqual(["id2", "id3", "id9"]);
  });

  it("mixes replay into the nightly batch", () => {
    const pairs = Array.from({ length: 10 }, (_, i) => ({
      ...makePair("edit", prompt, "c" + i + " x", "r" + i, i, "p" + i)!, trainedAt: i < 6 ? 100 : undefined,
    }));
    const batch = nightlyBatch(pairs, 64, 0.5);
    expect(batch.filter((p) => !p.trainedAt).length).toBe(4);
    expect(batch.filter((p) => p.trainedAt).length).toBe(2);
  });

  it("turns kept regenerations into pairs", () => {
    let n = 0;
    const pairs = regenerationPairs(prompt, ["first try", "second try is better", "third"], "second try is better", 5,
      () => `g${n++}`);
    expect(pairs.map((p) => p.rejected)).toEqual(["first try", "third"]);
    expect(pairs.every((p) => p.chosen === "second try is better" && p.source === "regenerate")).toBe(true);
  });
});
