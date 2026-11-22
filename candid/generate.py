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

Three resumable steps, each saved to the store as it finishes:
  prompts()   sample prompts (+ images) once            -> prompts/
  shards()    answer + select, cfg.shard_size at a time -> gen/shard-XXXXX.json
  assemble()  render chat-templated pairs               -> pairs/
"""

from __future__ import annotations

import json
import logging
import random
import re
from collections import Counter

from candid.config import Config
from candid.filters import detect_lang, is_clean, is_refusal, strip_think, violations
from candid.spec import DECLINE, JUDGE_PROMPT, SPEC

log = logging.getLogger("candid.generate")

CHUNK = 256


def _prep_image(img):
    """Decoded image for vLLM (stored images are already resized JPEGs)."""
    return None if img is None else img.convert("RGB")


def _user_msg(text: str, image) -> dict:
    if image is None:
        return {"role": "user", "content": text}
    return {"role": "user", "content": [{"type": "image_pil", "image_pil": image}, {"type": "text", "text": text}]}


def _chat(llm, convs, params, desc):
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


def select_pairs(rows, defaults, specs, rng: random.Random, judge=None):
    """Pure selection logic. Returns (pairs as (row_index, chosen, rejected), stats)."""
    stats = Counter()
    pairs, need_judge = [], []
    for i, (r, d, cands) in enumerate(zip(rows, defaults, specs)):
        b = r["bucket"]
        stats[f"{b}/total"] += 1
        if b == "hard_limit":
            decline = DECLINE[detect_lang(r["prompt"])]
            if d.strip() != decline:
                pairs.append((i, decline, d))
                stats[f"{b}/kept"] += 1
            continue
        good = [c for c in cands if is_clean(c)]
        if not good:
            stats[f"{b}/no_clean_spec"] += 1
            continue
        chosen = rng.choice(good)
        v = violations(d)
        if any(v.values()) or not d.strip():
            for k, hit in v.items():
                stats[f"{b}/default_{k}"] += int(hit)
            pairs.append((i, chosen, d))
            stats[f"{b}/kept"] += 1
        elif judge is not None:
            need_judge.append((i, chosen, d))
    if need_judge:
        wins = judge([rows[i] for i, _, _ in need_judge], [d for _, _, d in need_judge],
                     [c for _, c, _ in need_judge])
        for (i, c, d), w in zip(need_judge, wins):
            b = rows[i]["bucket"]
            stats[f"{b}/judged"] += 1
            if w:
                pairs.append((i, c, d))
                stats[f"{b}/kept"] += 1
                stats[f"{b}/judge_win"] += 1
    return pairs, stats


# ----- step 1: prompts --------------------------------------------------------

def prompts(cfg: Config, store) -> None:
    from datasets import Dataset, Features, Image, Value

    from candid.sources import build_prompts

    rows = build_prompts(cfg)
    log.info("prompts: %s", dict(Counter(r["bucket"] for r in rows)))
    rng = random.Random(cfg.seed)
    rng.shuffle(rows)  # mixes buckets across shards
    feats = Features({"id": Value("string"), "bucket": Value("string"), "lang": Value("string"),
                      "prompt": Value("string"), "image": Image()})
    ds = Dataset.from_dict({
        "id": [r["id"] for r in rows], "bucket": [r["bucket"] for r in rows], "lang": [r["lang"] for r in rows],
        "prompt": [r["prompt"] for r in rows], "image": [r.get("image") for r in rows],  # encoded bytes
    }, features=feats)
    ds.save_to_disk(store.local("prompts"))
    store.push("prompts", replace=True)


# ----- step 2: shards ---------------------------------------------------------

def shard_name(i: int) -> str:
    return f"gen/shard-{i:05d}.json"


def shards(cfg: Config, store) -> None:
    from datasets import load_from_disk

    store.pull("prompts")
    ds = load_from_disk(store.local("prompts"))
    n = (len(ds) + cfg.shard_size - 1) // cfg.shard_size
    todo = [i for i in range(n) if not store.exists(shard_name(i))]
    log.info("shards: %d total, %d to do", n, len(todo))
    if not todo:
        return

    from vllm import LLM, SamplingParams

    llm = LLM(model=cfg.base_model, max_model_len=cfg.gen_max_model_len, gpu_memory_utilization=cfg.gen_gpu_mem,
              limit_mm_per_prompt={"image": 1}, seed=cfg.seed)
    default_p = SamplingParams(temperature=cfg.gen_temperature, top_p=0.95, max_tokens=cfg.gen_max_tokens)
    spec_p = SamplingParams(n=cfg.gen_samples, temperature=cfg.gen_temperature, top_p=0.95,
                            max_tokens=cfg.gen_max_tokens)
    judge = (lambda rows, a, b: _judge(llm, rows, a, b)) if cfg.use_self_judge else None

    for i in todo:
        lo, hi = i * cfg.shard_size, min(len(ds), (i + 1) * cfg.shard_size)
        rows = [dict(ex) for ex in ds.select(range(lo, hi))]
        for r in rows:
            r["image"] = _prep_image(r["image"])
        defaults = [o[0] for o in _chat(llm, [[_user_msg(r["prompt"], r["image"])] for r in rows], default_p,
                                        f"shard {i} default")]
        specs = _chat(llm, [[{"role": "system", "content": SPEC}, _user_msg(r["prompt"], r["image"])] for r in rows],
                      spec_p, f"shard {i} spec")
        pairs, stats = select_pairs(rows, defaults, specs, random.Random(cfg.seed + i), judge)
        store.write_json(shard_name(i), {
            "pairs": [{"idx": lo + j, "chosen": c, "rejected": d} for j, c, d in pairs],
            "stats": dict(stats),
        })
        log.info("shard %d/%d: %d pairs", i + 1, n, len(pairs))
        store.update_state(progress=f"generate shard {i + 1}/{n}")


# ----- step 3: assemble -------------------------------------------------------

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


def assemble(cfg: Config, store) -> dict:
    from datasets import Dataset, Features, Image, Sequence, Value, load_from_disk
    from transformers import AutoProcessor

    from candid.sources import image_preference_pairs

    store.pull("prompts")
    store.pull("gen")
    # Keep images as encoded bytes here; decoding thousands of them at once exhausts RAM.
    ds = load_from_disk(store.local("prompts")).cast_column("image", Image(decode=False))
    n = (len(ds) + cfg.shard_size - 1) // cfg.shard_size
    stats, items = Counter(), []
    for i in range(n):
        shard = store.read_json(shard_name(i))
        if shard is None:
            raise RuntimeError(f"missing {shard_name(i)}")
        stats.update(shard["stats"])
        for p in shard["pairs"]:
            ex = ds[p["idx"]]
            items.append((ex["prompt"], ex["image"], p["chosen"], p["rejected"], ex["bucket"], ex["id"]))

    for j, p in enumerate(image_preference_pairs(cfg)):
        if is_clean(p["chosen"]) and not is_refusal(p["chosen"]):
            items.append((p["prompt"], p["image"], p["chosen"], p["rejected"], "image_pref",
                          f"image_pref-{j}"))
            stats["image_pref/kept"] += 1

    processor = AutoProcessor.from_pretrained(cfg.base_model)
    recs = {"images": [], "prompt": [], "chosen": [], "rejected": [], "bucket": [], "id": []}
    for prompt, image, chosen, rejected, bucket, rid in items:
        has_img = image is not None
        prompt_s, chosen_s = _render(processor, prompt, has_img, chosen)
        _, rejected_s = _render(processor, prompt, has_img, rejected)
        recs["images"].append([image] if has_img else [])
        recs["prompt"].append(prompt_s)
        recs["chosen"].append(chosen_s)
        recs["rejected"].append(rejected_s)
        recs["bucket"].append(bucket)
        recs["id"].append(rid)

    feats = Features({"images": Sequence(Image()), "prompt": Value("string"), "chosen": Value("string"),
                      "rejected": Value("string"), "bucket": Value("string"), "id": Value("string")})
    pairs = Dataset.from_dict(recs, features=feats).shuffle(seed=cfg.seed)
    pairs.save_to_disk(store.local("pairs"))
    store.push("pairs", replace=True)

    rng = random.Random(cfg.seed)
    sample = rng.sample(items, min(200, len(items)))
    with open(store.local("pairs_sample.jsonl"), "w") as f:
        for prompt, _, chosen, rejected, bucket, _ in sample:
            f.write(json.dumps({"bucket": bucket, "prompt": prompt, "chosen": chosen, "rejected": rejected},
                               ensure_ascii=False) + "\n")
    store.push("pairs_sample.jsonl")
    out = {"pairs": len(items), **dict(sorted(stats.items()))}
    store.write_json("gen_stats.json", out)
    log.info("assembled: %s", out)
    return out


# ----- diagnostics ------------------------------------------------------------

_SOFT_REFUSAL = re.compile(r"\b(?:I cannot|I can't|I can not|I won't|I'm unable|I am unable|I must decline)\b",
                           re.IGNORECASE)


def _clip(s: str, n: int) -> str:
    s = " ".join((s or "").split())
    return s if len(s) <= n else s[: n - 1] + "…"


def data_report(cfg: Config, store, samples_per_bucket: int = 3) -> dict:
    """Per-bucket pair counts and a few sample pairs, as short log lines (the log is all we can read)."""
    from datasets import Image, load_from_disk

    ds = load_from_disk(store.local("prompts")).cast_column("image", Image(decode=False))
    buckets, prompts = ds["bucket"], ds["prompt"]
    n = (len(ds) + cfg.shard_size - 1) // cfg.shard_size
    stats, examples = Counter(), {}
    soft = Counter()
    for i in range(n):
        shard = store.read_json(shard_name(i)) or {"pairs": [], "stats": {}}
        stats.update(shard["stats"])
        for p in shard["pairs"]:
            b = buckets[p["idx"]]
            if _SOFT_REFUSAL.search(p["chosen"]):
                soft[b] += 1
            examples.setdefault(b, [])
            if len(examples[b]) < samples_per_bucket:
                examples[b].append((prompts[p["idx"]], p["chosen"], p["rejected"]))
    report = {}
    for b in sorted({k.split("/")[0] for k in stats}):
        row = {k.split("/", 1)[1]: v for k, v in stats.items() if k.startswith(b + "/")}
        row["soft_refusal_in_chosen"] = soft[b]
        report[b] = row
        print(f"[report] {b} " + " ".join(f"{k}={v}" for k, v in sorted(row.items())), flush=True)
    for b, rows in examples.items():
        for prompt, chosen, rejected in rows:
            print(f"[report] sample {b} PROMPT: {_clip(prompt, 300)}", flush=True)
            print(f"[report] sample {b} CHOSEN: {_clip(chosen, 400)}", flush=True)
            print(f"[report] sample {b} REJECTED: {_clip(rejected, 400)}", flush=True)
    return report
