import { describe, expect, it } from "vitest";

import { evaluate, runCalc } from "../src/llm/calc";
import { deepThink, extractFinal, normalizeAnswer, vote } from "../src/llm/deepThink";
import type { CompleteFn } from "../src/llm/toolLoop";

describe("calculator", () => {
  it("evaluates arithmetic with precedence, functions and constants", () => {
    expect(evaluate("2 + 3 * 4")).toBe(14);
    expect(evaluate("(2 + 3) * 4")).toBe(20);
    expect(evaluate("2^3^2")).toBe(512); // right-associative
    expect(evaluate("-2^2")).toBe(-4);
    expect(evaluate("10 % 4 + 5!")).toBe(122);
    expect(evaluate("sqrt(16) + max(1, 7, 3)")).toBe(11);
    expect(evaluate("2pi")).toBeCloseTo(2 * Math.PI);
    expect(evaluate("1250 × 0.18 ÷ 12")).toBeCloseTo(18.75);
    expect(evaluate("1.5e3 / 3")).toBe(500);
  });

  it("rejects anything that is not arithmetic", () => {
    expect(() => evaluate("process.exit(1)")).toThrow();
    expect(() => evaluate("2 +")).toThrow();
    expect(() => evaluate("1/0")).toThrow(/finite/);
    expect(runCalc('{"expression":"alert(1)"}')).toMatch(/tool error/);
    expect(runCalc('{"expression":"(1250*0.18)/12"}')).toBe("(1250*0.18)/12 = 18.75");
  });
});

describe("answers and voting", () => {
  it("finds the committed answer", () => {
    expect(extractFinal("work...\nAnswer: 42")).toBe("42");
    expect(extractFinal("<think>Answer: 1</think>so \\boxed{7}")).toBe("7");
    expect(extractFinal("Cevap: 18,75 TL")).toBe("18,75 TL");
    expect(extractFinal("no final line")).toBeNull();
  });

  it("normalizes equivalent answers to one key", () => {
    expect(normalizeAnswer("1,250.00 TL")).toBe(normalizeAnswer("1250"));
    expect(normalizeAnswer("**x = 3**.")).toBe(normalizeAnswer("3"));
    expect(normalizeAnswer("Paris")).toBe(normalizeAnswer("paris."));
  });

  it("picks the majority", () => {
    const a = (key: string | null) => ({ text: key ?? "-", answer: key, key });
    const v = vote([a("#3"), a("#4"), a("#3"), a(null)]);
    expect(v).toMatchObject({ key: "#3", support: 2, total: 4 });
    expect(vote([a(null)]).winner?.text).toBe("-");
  });
});

describe("deepThink", () => {
  it("uses the calculator, votes, and stops early on agreement", async () => {
    const answers = ["Answer: 18.75", "Answer: 18.75", "Answer: 20"];
    let calls = 0;
    const tools: string[] = [];
    const complete: CompleteFn = async ({ messages, tools: offered }) => {
      calls++;
      const last = messages[messages.length - 1];
      if (last.role === "user") {
        expect(String(last.content)).toMatch(/calculate tool/);
        expect((offered ?? []).length).toBe(1);
        return { content: "", text: "", tool_calls: [{ type: "function", id: "t", function: { name: "calculate",
          arguments: '{"expression":"1250*0.18/12"}' } }] };
      }
      tools.push(String(last.content));
      return { content: answers.shift()!, text: "" };
    };
    const res = await deepThink({ complete, messages: [{ role: "user", content: "1250 TL'nin %18'inin 12'de biri?" }],
      attempts: 3 });
    expect(res).toMatchObject({ support: 2, total: 2, agreed: true });
    expect(res.answer).toBe("Answer: 18.75");
    expect(calls).toBe(4); // two attempts, each: tool call + answer; the third was not needed
    expect(tools[0]).toBe("1250*0.18/12 = 18.75");
  });
});
