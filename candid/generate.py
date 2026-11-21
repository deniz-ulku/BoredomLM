"""Build preference pairs from the base model's own answers.

For every prompt the base model answers twice:
  * default: no system prompt (how it behaves today)            -> candidate "rejected"
  * spec:    with candid.spec.SPEC as system prompt, best-of-n -> candidate "chosen"

A pair is kept only when the spec answer is clean and actually better: either the
default answer breaks a rule that the regex checks catch (refusal, preaching, filler,
emoji), or the base model, acting as judge, prefers the spec answer in both A/B orders.

Both sides come from the same model, so the preference signal is about behaviour,
not knowledge. That is what keeps intelligence where it was (no abliteration, no
weight surgery).

Usage (on the GPU machine):  python -m candid.generate
"""

from __future__ import annotations

import json
import logging
import random
from collections import Counter

from candid.config import Config
from candid.filters import detect_lang, is_clean, is_refusal, strip_think, violations
from candid.sources import build_prompts, image_preference_pairs
from candid.spec import DECLINE, JUDGE_PROMPT, SPEC

log = logging.getLogger("candid.generate")

MAX_IMAGE_SIDE = 1536
CHUNK = 256


def _prep_image(img):
    if img is None:
        return None
    img = img.convert("RGB")
    w, h = img.size
    s = MAX_IMAGE_SIDE / max(w, h)
    if s < 1:
        img = img.resize((max(1, int(w * s)), max(1, int(h * s))))
    return img


def _user_msg(text: str, image) -> dict:
    if image is None:
        return {"role": "user", "content": text}
    return {"role": "user", "content": [{"type": "image_pil", "image_pil": image}, {"type": "text", "text": text}]}


def _chat(llm, convs, params, desc):
    from vllm import SamplingParams  # noqa: F401  (imported for type clarity on the GPU box)

    outs = []
    for i in range(0, len(convs), CHUNK):
        log.info("%s: %d/%d", desc, i, len(convs))
        res = llm.chat(convs[i:i + CHUNK], params, use_tqdm=False,
                       chat_template_kwargs={"enable_thinking": False})
        outs += [[strip_think(o.text) for o in r.outputs] for r in res]
    return outs


def _judge(llm, rows, a_texts, b_texts) -> list[bool]:
    """True where the judge prefers B over A in both presentation orders."""
    from vllm import SamplingParams

    p = SamplingParams(temperature=0.0, max_tokens=2)
    convs_ab, convs_ba = [], []
    for r, a, b in zip(rows, a_texts, b_texts):
        convs_ab.append([_user_msg(JUDGE_PROMPT.format(prompt=r["prompt"], a=a, b=b), r["image"])])
        convs_ba.append([_user_msg(JUDGE_PROMPT.format(prompt=r["prompt"], a=b, b=a), r["image"])])
    ab = _chat(llm, convs_ab, p, "judge A/B")
    ba = _chat(llm, convs_ba, p, "judge B/A")
    return [x[0].strip().upper().startswith("B") and y[0].strip().upper().startswith("A") for x, y in zip(ab, ba)]


def _render(processor, prompt: str, has_image: bool, answer: str) -> tuple[str, str]:
    """Explicit (pre-templated) prompt/completion strings, non-thinking mode."""
    content = [{"type": "image"}, {"type": "text", "text": prompt}] if has_image else prompt
    user = [{"role": "user", "content": content}]
    kw = {"tokenize": False, "enable_thinking": False}
    prompt_s = processor.apply_chat_template(user, add_generation_prompt=True, **kw)
    full = processor.apply_chat_template(user + [{"role": "assistant", "content": answer}], **kw)
    if full.startswith(prompt_s):
        return prompt_s, full[len(prompt_s):]
    tok = getattr(processor, "tokenizer", processor)
    eot = "<|im_end|>" if "<|im_end|>" in tok.get_vocab() else tok.eos_token
    return prompt_s, answer + eot + "\n"


def run(cfg: Config) -> dict:
    from datasets import Dataset, Features, Image, Sequence, Value
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    random.seed(cfg.seed)
    rows = build_prompts(cfg)
    for r in rows:
        r["image"] = _prep_image(r.get("image"))
    log.info("prompts: %s", dict(Counter(r["bucket"] for r in rows)))

    llm = LLM(model=cfg.base_model, max_model_len=cfg.gen_max_model_len, gpu_memory_utilization=cfg.gen_gpu_mem,
              limit_mm_per_prompt={"image": 1}, seed=cfg.seed)
    default_p = SamplingParams(temperature=cfg.gen_temperature, top_p=0.95, max_tokens=cfg.gen_max_tokens)
    spec_p = SamplingParams(n=cfg.gen_samples, temperature=cfg.gen_temperature, top_p=0.95,
                            max_tokens=cfg.gen_max_tokens)

    defaults = [o[0] for o in _chat(llm, [[_user_msg(r["prompt"], r["image"])] for r in rows], default_p, "default")]
    specs = _chat(llm, [[{"role": "system", "content": SPEC}, _user_msg(r["prompt"], r["image"])] for r in rows],
                  spec_p, "spec")

    stats = Counter()
    pairs, need_judge = [], []
    for r, d, cands in zip(rows, defaults, specs):
        stats[f"{r['bucket']}/total"] += 1
        if r["bucket"] == "hard_limit":
            decline = DECLINE[detect_lang(r["prompt"])]
            if d.strip() != decline:
                pairs.append((r, decline, d))
                stats[f"{r['bucket']}/kept"] += 1
            continue
        good = [c for c in cands if is_clean(c)]
        if not good:
            stats[f"{r['bucket']}/no_clean_spec"] += 1
            continue
        chosen = random.choice(good)
        v = violations(d)
        if any(v.values()) or not d.strip():
            for k, hit in v.items():
                stats[f"{r['bucket']}/default_{k}"] += int(hit)
            pairs.append((r, chosen, d))
            stats[f"{r['bucket']}/kept"] += 1
        elif cfg.use_self_judge:
            need_judge.append((r, chosen, d))

    if need_judge:
        wins = _judge(llm, [x[0] for x in need_judge], [x[2] for x in need_judge], [x[1] for x in need_judge])
        for (r, c, d), w in zip(need_judge, wins):
            stats[f"{r['bucket']}/judged"] += 1
            if w:
                pairs.append((r, c, d))
                stats[f"{r['bucket']}/kept"] += 1
                stats[f"{r['bucket']}/judge_win"] += 1
    del llm

    for p in image_preference_pairs(cfg):
        if is_clean(p["chosen"]) and not is_refusal(p["chosen"]):
            pairs.append(({**p, "image": _prep_image(p["image"])}, p["chosen"], p["rejected"]))
            stats["image_pref/kept"] += 1

    processor = AutoProcessor.from_pretrained(cfg.base_model)
    recs = {"images": [], "prompt": [], "chosen": [], "rejected": [], "bucket": [], "id": []}
    for r, c, d in pairs:
        has_img = r["image"] is not None
        prompt_s, chosen_s = _render(processor, r["prompt"], has_img, c)
        _, rejected_s = _render(processor, r["prompt"], has_img, d)
        recs["images"].append([r["image"]] if has_img else [])
        recs["prompt"].append(prompt_s)
        recs["chosen"].append(chosen_s)
        recs["rejected"].append(rejected_s)
        recs["bucket"].append(r["bucket"])
        recs["id"].append(r["id"])

    feats = Features({"images": Sequence(Image()), "prompt": Value("string"), "chosen": Value("string"),
                      "rejected": Value("string"), "bucket": Value("string"), "id": Value("string")})
    ds = Dataset.from_dict(recs, features=feats).shuffle(seed=cfg.seed)
    ds.save_to_disk(cfg.path("pairs"))

    with open(cfg.path("pairs_sample.jsonl"), "w") as f:
        for r, c, d in random.sample(pairs, min(200, len(pairs))):
            f.write(json.dumps({"bucket": r["bucket"], "prompt": r["prompt"], "chosen": c, "rejected": d},
                               ensure_ascii=False) + "\n")
    out = {"pairs": len(pairs), **dict(sorted(stats.items()))}
    with open(cfg.path("gen_stats.json"), "w") as f:
        json.dump(out, f, indent=2)
    log.info("generation done: %s", out)
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    run(Config.from_env())
