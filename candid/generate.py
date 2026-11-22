"""Build preference pairs from the base model's own answers (v1 recipe).

Style pairs (general, Turkish and image prompts):
  rejected = the base model's default answer, when it breaks a style rule (emoji, filler,
             preaching, exclamation marks)
  chosen   = the same answer rewritten by the model under EDIT_PROMPT: same content, only
             the packaging removed. A rewrite that drops too much text or still breaks a rule
             is discarded.
Refusal pairs (OR-Bench-80k prompts: seemingly sensitive, actually benign):
  every prompt is screened with a short default answer; only the ones the base refuses go on.
  rejected = the full default (refusing) answer
  chosen   = a compliant answer written with SPEC + PERMISSION as system prompt (best of n)

Every target comes from the same model, so the preference is about behaviour, not knowledge.
v0 also used the base model as a judge between two different answers; it favoured short
answers even when they were wrong, which cost capability, so v1 has no judge.

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
from candid.filters import is_clean, is_refusal, strip_style, strip_think, violations
from candid.spec import EDIT_PROMPT, PERMISSION, SPEC

log = logging.getLogger("candid.generate")

CHUNK = 256
SCREENED = "over_refusal"


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


# ----- pure selection logic (unit-tested) --------------------------------------

def needs(default: str) -> str:
    """What a default answer needs: 'spec' (it refuses), 'edit' (style only) or 'none'."""
    v = violations(default)
    if v["refusal"] or not default.strip():
        return "spec"
    if any(v.values()):
        return "edit"
    return "none"


def clean_edit(text: str) -> str:
    t = strip_think(text).strip()
    t = re.sub(r"^\s*(<<<|>>>)\s*|\s*(<<<|>>>)\s*$", "", t)
    return t.strip()


def accept_edit(original: str, edited: str, min_ratio: float) -> str | None:
    """Reason the rewrite is rejected, or None if it is a good chosen answer."""
    if not edited:
        return "empty"
    if not is_clean(edited) or is_refusal(edited):
        return "unclean"
    if len(edited) < min_ratio * len(original):
        return "too_short"
    return None


def pick_compliant(candidates: list[str]) -> str | None:
    for c in candidates:
        if is_clean(c) and not is_refusal(c):
            return c
    return None


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
    full = SamplingParams(temperature=cfg.gen_temperature, top_p=0.95, max_tokens=cfg.gen_max_tokens)
    short = SamplingParams(temperature=cfg.gen_temperature, top_p=0.95, max_tokens=cfg.screen_max_tokens)
    many = SamplingParams(n=cfg.gen_samples, temperature=cfg.gen_temperature, top_p=0.95,
                          max_tokens=cfg.gen_max_tokens)
    edit_p = SamplingParams(temperature=0.3, top_p=0.9, max_tokens=cfg.gen_max_tokens)

    for i in todo:
        lo, hi = i * cfg.shard_size, min(len(ds), (i + 1) * cfg.shard_size)
        rows = [dict(ex) for ex in ds.select(range(lo, hi))]
        for r in rows:
            r["image"] = _prep_image(r["image"])
        pairs, stats, rejects = _shard_pairs(llm, rows, cfg, full, short, many, edit_p, f"shard {i}")
        store.write_json(shard_name(i), {
            "pairs": [{"idx": lo + j, "chosen": c, "rejected": d, "kind": k} for j, c, d, k in pairs],
            "stats": dict(stats),
            "rejected_edits": rejects,
        })
        log.info("shard %d/%d: %d pairs", i + 1, n, len(pairs))
        store.update_state(progress=f"generate shard {i + 1}/{n}")


def _shard_pairs(llm, rows, cfg, full, short, many, edit_p, tag):
    stats = Counter()
    for r in rows:
        stats[f"{r['bucket']}/total"] += 1
    msg = lambda r: [_user_msg(r["prompt"], r["image"])]

    # 1. Screened prompts get a short answer first; only refusals go on to a full one.
    screened = [j for j, r in enumerate(rows) if r["bucket"] == SCREENED]
    shorts = _chat(llm, [msg(rows[j]) for j in screened], short, f"{tag} screen")
    refused = [j for j, o in zip(screened, shorts) if is_refusal(o[0])]
    stats.update({f"{SCREENED}/screened_refused": len(refused)})
    full_idx = [j for j, r in enumerate(rows) if r["bucket"] != SCREENED] + refused
    defaults = dict(zip(full_idx, (o[0] for o in _chat(llm, [msg(rows[j]) for j in full_idx], full,
                                                         f"{tag} default"))))

    # 2. Decide per prompt, then batch the follow-up generations.
    want = {j: needs(d) for j, d in defaults.items()}
    for j, w in want.items():
        stats[f"{rows[j]['bucket']}/need_{w}"] += 1
        for k, hit in violations(defaults[j]).items():
            stats[f"{rows[j]['bucket']}/default_{k}"] += int(hit)
    spec_idx = [j for j, w in want.items() if w == "spec"]
    edit_idx = [j for j, w in want.items() if w == "edit"]
    # Emojis, exclamation marks and filler are removed mechanically (content kept by construction);
    # only preaching needs the model to rewrite sentences.
    model_edit = [j for j in edit_idx if violations(defaults[j])["preachy"]]

    def spec_conv(j):
        system = SPEC + (PERMISSION if rows[j]["bucket"] == SCREENED else "")
        return [{"role": "system", "content": system}, *msg(rows[j])]

    specs = dict(zip(spec_idx, _chat(llm, [spec_conv(j) for j in spec_idx], many, f"{tag} spec")))
    rewritten = dict(zip(model_edit, (clean_edit(o[0]) for o in _chat(
        llm, [[{"role": "user", "content": EDIT_PROMPT.format(answer=defaults[j])}] for j in model_edit], edit_p,
        f"{tag} edit"))))
    edits = {j: strip_style(rewritten.get(j, defaults[j])) for j in edit_idx}
    for j in model_edit:
        stats[f"{rows[j]['bucket']}/model_edit"] += 1

    pairs, rejects = [], []
    for j in spec_idx:
        b, chosen = rows[j]["bucket"], pick_compliant(specs[j])
        if chosen is None:
            stats[f"{b}/no_compliant"] += 1
            continue
        pairs.append((j, chosen, defaults[j], "refusal"))
        stats[f"{b}/kept_refusal"] += 1
    for j in edit_idx:
        # Only a model rewrite can lose content; the mechanical cleanup cannot.
        ratio = cfg.min_edit_ratio if j in rewritten else 0.0
        b, why = rows[j]["bucket"], accept_edit(defaults[j], edits[j], ratio)
        if why:
            stats[f"{b}/edit_{why}"] += 1
            if len(rejects) < 6:
                bad = [k for k, hit in violations(edits[j]).items() if hit]
                rejects.append({"bucket": b, "why": why, "violations": bad, "edited": edits[j][:600]})
            continue
        pairs.append((j, edits[j], defaults[j], "style"))
        stats[f"{b}/kept_style"] += 1
    return pairs, stats, rejects


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
    log.info("assembled: %d pairs", len(items))
    data_report(cfg, store)
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
    for i in range(n):
        for r in (store.read_json(shard_name(i)) or {}).get("rejected_edits", [])[:2]:
            print(f"[report] rejected edit {r['bucket']} why={r['why']} violations={r['violations']} "
                  f"TEXT: {_clip(r['edited'], 300)}", flush=True)
    for b, rows in examples.items():
        for prompt, chosen, rejected in rows:
            print(f"[report] sample {b} PROMPT: {_clip(prompt, 300)}", flush=True)
            print(f"[report] sample {b} CHOSEN: {_clip(chosen, 400)}", flush=True)
            print(f"[report] sample {b} REJECTED: {_clip(rejected, 400)}", flush=True)
    return report
