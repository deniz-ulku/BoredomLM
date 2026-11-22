// A calculator the model can call to check arithmetic. Parses and evaluates the expression
// itself (no eval / Function), so it is safe to run model-written input on the phone.
// Numbers use "." as the decimal point; "," only separates function arguments.

const FUNCS: Record<string, (...a: number[]) => number> = {
  sqrt: Math.sqrt, cbrt: Math.cbrt, abs: Math.abs, floor: Math.floor, ceil: Math.ceil, round: Math.round,
  exp: Math.exp, ln: Math.log, log: Math.log10, log10: Math.log10, log2: Math.log2,
  sin: Math.sin, cos: Math.cos, tan: Math.tan, asin: Math.asin, acos: Math.acos, atan: Math.atan,
  min: Math.min, max: Math.max, pow: Math.pow,
};
const CONSTS: Record<string, number> = { pi: Math.PI, e: Math.E };

type Tok = { t: "num"; v: number } | { t: "id"; v: string } | { t: "op"; v: string };

function lex(src: string): Tok[] {
  const s = src.replace(/×/g, "*").replace(/÷/g, "/").replace(/−/g, "-").replace(/\*\*/g, "^");
  const out: Tok[] = [];
  let i = 0;
  while (i < s.length) {
    const c = s[i];
    if (/\s/.test(c)) { i++; continue; }
    const num = s.slice(i).match(/^(\d+(?:\.\d+)?|\.\d+)(e[+-]?\d+)?/i);
    if (num) {
      out.push({ t: "num", v: Number(num[0]) });
      i += num[0].length;
      continue;
    }
    const id = s.slice(i).match(/^[a-z_][a-z0-9_]*/i);
    if (id) { out.push({ t: "id", v: id[0].toLowerCase() }); i += id[0].length; continue; }
    if ("+-*/^%(),!".includes(c)) { out.push({ t: "op", v: c }); i++; continue; }
    throw new Error(`unexpected character "${c}"`);
  }
  return out;
}

/** Evaluate an arithmetic expression: + - * / ^ % ! ( ) and the functions/constants above. */
export function evaluate(expr: string): number {
  const toks = lex(expr);
  let p = 0;
  const peek = () => toks[p];
  const eat = (v?: string) => {
    const tok = toks[p];
    if (!tok || (v !== undefined && tok.v !== v)) throw new Error(`expected "${v ?? "value"}"`);
    p++;
    return tok;
  };
  const isOp = (v: string) => peek()?.t === "op" && peek()!.v === v;

  function expression(): number {
    let v = term();
    while (isOp("+") || isOp("-")) v = eat().v === "+" ? v + term() : v - term();
    return v;
  }
  function term(): number {
    let v = unary();
    for (;;) {
      if (isOp("*")) { eat(); v *= unary(); }
      else if (isOp("/")) { eat(); v /= unary(); }
      else if (isOp("%")) { eat(); v %= unary(); }
      else if (peek()?.t === "num" || peek()?.t === "id" || isOp("(")) v *= unary(); // implicit: 2(3+1), 2pi
      else return v;
    }
  }
  function unary(): number {
    if (isOp("-")) { eat(); return -unary(); }
    if (isOp("+")) { eat(); return unary(); }
    return power();
  }
  function power(): number {
    const base = postfix();
    if (isOp("^")) { eat(); return Math.pow(base, unary()); }
    return base;
  }
  function postfix(): number {
    let v = primary();
    while (isOp("!")) {
      eat();
      if (!Number.isInteger(v) || v < 0 || v > 170) throw new Error("factorial needs an integer 0..170");
      let f = 1;
      for (let k = 2; k <= v; k++) f *= k;
      v = f;
    }
    return v;
  }
  function primary(): number {
    const tok = peek();
    if (!tok) throw new Error("unexpected end");
    if (tok.t === "num") { p++; return tok.v; }
    if (tok.t === "op" && tok.v === "(") { eat("("); const v = expression(); eat(")"); return v; }
    if (tok.t === "id") {
      p++;
      if (tok.v in CONSTS && !isOp("(")) return CONSTS[tok.v];
      const fn = FUNCS[tok.v];
      if (!fn) throw new Error(`unknown name "${tok.v}"`);
      eat("(");
      const args = [expression()];
      while (isOp(",")) { eat(","); args.push(expression()); }
      eat(")");
      return fn(...args);
    }
    throw new Error(`unexpected "${tok.v}"`);
  }

  const v = expression();
  if (p < toks.length) throw new Error(`unexpected "${toks[p].v}"`);
  if (!Number.isFinite(v)) throw new Error("result is not a finite number");
  return v;
}

export function formatNumber(v: number): string {
  if (Number.isInteger(v)) return String(v);
  return String(Number(v.toPrecision(12)));
}

export const CALC_TOOL = {
  type: "function",
  function: {
    name: "calculate",
    description: "Evaluate an arithmetic expression exactly. Use it for every non-trivial calculation.",
    parameters: {
      type: "object",
      properties: { expression: { type: "string", description: "e.g. (1250 * 0.18) / 12 or sqrt(2) ^ 3" } },
      required: ["expression"],
    },
  },
} as const;

export function runCalc(argsJson: string): string {
  try {
    const { expression } = JSON.parse(argsJson || "{}") as { expression?: string };
    if (!expression) return "[tool error: empty expression]";
    return `${expression} = ${formatNumber(evaluate(expression))}`;
  } catch (e) {
    return `[tool error: ${(e as Error).message}]`;
  }
}
