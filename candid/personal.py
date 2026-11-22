"""Server-side simulation of on-device learning (docs/ON_DEVICE_LEARNING.md, section 6).

Three synthetic users each give ~50 corrections to Candid. A phone-sized LoRA (rank 8,
512 tokens) is trained on top of the shipped model, once per user, like one night on the
phone. Reported per user and training setting: training time, how often held-out answers now
match what the user wants (before vs after), and GSM8K accuracy before vs after.

First run (lr 2e-5, NLL weight 0.2): nothing moved (terse 0 -> 0, formal_tr 0 -> 0), GSM8K
held. The pairs were learned (reward accuracy 1.0) but greedy answers did not change, so
the settings below are stronger.

Second run (lr 1e-4 / 2e-4, NLL 1.0): style and facts moved (terse 0 -> 100, formal_tr
0 -> 73-80, facts 7 -> 73-100), but the terse user's adapter cut GSM8K from 90 to 80: it
stopped showing work even when the prompt asked for it. Anchor pairs (below) target that.

    CANDID_MODE=personal on the machine (see remote/run.sh), or python -m candid.personal
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import shutil
import time
from dataclasses import dataclass, field

from candid.config import Config
from candid.filters import detect_lang, strip_think

log = logging.getLogger("candid.personal")


# ----- synthetic users ----------------------------------------------------------

@dataclass
class Persona:
    name: str
    instruction: str          # how this user wants answers (used to write the "corrected" answers)
    prompts: list[str]        # questions the user asks; split into train / held-out
    check: str                # metric name, see persona_score
    facts: list[str] = field(default_factory=list)


GENERAL = [
    "How do I make my laptop battery last longer?", "What is a good way to learn a new language?",
    "Explain what inflation is.", "How do I fix a slow phone?", "What should I pack for a weekend trip?",
    "How does a credit score work?", "What is the difference between a virus and bacteria?",
    "How do I start running if I have never exercised?", "Why do cats purr?", "How do I write a good email subject line?",
    "What is machine learning?", "How can I sleep better?", "How do I remove a coffee stain?",
    "What are some tips for a job interview?", "How do vaccines work?", "What is the best way to save money each month?",
    "How do I keep houseplants alive?", "Explain how a mortgage works.", "What is a healthy breakfast?",
    "How do I back up my photos?", "Why is the ocean salty?", "How do I negotiate a salary?",
    "What is compound interest?", "How do I make cold brew coffee?", "How do noise cancelling headphones work?",
    "What is a good first car?", "How can I be more productive working from home?", "How do airplanes stay in the air?",
    "What is the difference between weather and climate?", "How do I clean my keyboard?", "What makes bread rise?",
    "How do I choose a good password?", "What is an index fund?", "How do I deal with procrastination?",
    "Why do we have leap years?", "How do I learn to cook?", "What is a VPN?", "How do I stop my glasses fogging up?",
    "How do solar panels work?", "What should I do if my phone gets wet?", "How do I make friends in a new city?",
    "What is the stock market?", "How do I get better at public speaking?", "Why is sleep important?",
    "How do I organize my closet?", "What is blockchain?", "How do I keep my car in good shape?",
    "How do I take better photos with my phone?", "What is the best way to study for an exam?",
    "How do I reduce food waste?", "How does GPS work?", "What should I look for when renting a flat?",
    "How do I calm down before a big meeting?", "Why do leaves change colour?", "How do I start a small garden?",
    "What is a calorie?", "How do I get rid of hiccups?", "How do I pick a good watermelon?",
    "What is the difference between RAM and storage?", "How do I learn to type faster?",
]

FACT_PROMPTS = [
    "Suggest a quick snack for this afternoon.", "What should I cook for dinner tonight?", "Plan my Saturday for me.",
    "Recommend a weekend activity near where I live.", "What should I bring to a friend's birthday party?",
    "Give me ideas for a healthy lunch.", "What can I do this evening to relax?", "Suggest a gift for myself under 500 TL.",
    "What's a good dessert I can make?", "Plan a short trip for next weekend.", "What should I eat before my workout?",
    "Any ideas for a fun Sunday morning?", "Recommend a café-style breakfast I can make at home.",
    "What should I get at the supermarket this week?", "Suggest a picnic menu.", "What should I do on a rainy day?",
    "Plan a dinner for two.", "What can I bake this weekend?", "Suggest a snack for a long train ride.",
    "What should I order at a Thai restaurant?", "Give me a morning routine idea.", "What could I cook with friends?",
    "Recommend a smoothie recipe.", "What should I do after work today?", "Suggest a cheap date idea.",
    "Plan a holiday morning.", "What should I make for a potluck?", "Any ideas for a lazy Sunday dinner?",
    "Suggest a high-protein snack.", "What's a good street food to try?", "Plan a movie night.",
    "What should I eat when I'm sick?", "Recommend something to do outdoors this week.", "What should I pack for lunch?",
    "Suggest a breakfast for a busy morning.", "What should I cook for my parents?", "Give me a healthy dessert idea.",
    "What can I do with my evening besides TV?", "Suggest a recipe with chocolate.", "Plan a beach day.",
    "Recommend a snack for a hike.", "What should I bring to a picnic?", "Suggest a cookie recipe.",
    "What can I make with what's usually in the fridge?", "Plan a day off.", "Recommend a salad.",
    "What should I eat at the airport?", "Suggest a pastry to try.", "Plan a small dinner party.",
    "What's a good after-school snack?", "Recommend a soup for winter.", "Suggest a weekend breakfast.",
    "What should I do for my birthday?", "Recommend a trail mix.", "Plan a cosy night in.",
    "What should I make for a bake sale?", "Suggest a lunch I can eat at my desk.", "Recommend an Asian dish to cook.",
    "What sauce goes with noodles?", "Suggest a dessert for guests.",
]

PERSONAS = [
    Persona("terse", "Answer in all lowercase letters, in at most two short sentences, with no lists or headings.",
            GENERAL, "terse"),
    Persona("formal_tr", "Her zaman Türkçe cevap ver, kullanıcıya 'siz' diye hitap et ve resmi, kibar bir dil kullan.",
            GENERAL, "formal_tr"),
    Persona("facts", "Take these facts about the user into account whenever they matter.", FACT_PROMPTS, "facts",
            facts=["I am severely allergic to peanuts.", "I live in İzmir.", "I have a cat named Pamuk."]),
]


def persona_score(check: str, answer: str) -> bool:
    """Does an answer match what this user wants?"""
    a = strip_think(answer).strip()
    if check == "terse":
        sentences = [s for s in re.split(r"(?<=[.!?])\s+", a) if s.strip()]
        return a == a.lower() and len(sentences) <= 2 and not re.search(r"^\s*([-*]|\d+\.|#)", a, re.MULTILINE)
    if check == "formal_tr":
        # "siz" forms anywhere in a word: ayarlar-ınız-ı, etme-niz-i, eder-siniz
        return detect_lang(a) == "tr" and bool(re.search(r"\b(siz|size|sizin|sizi|sizce)\b|iniz|ınız|unuz|ünüz",
                                                         a.lower()))
    if check == "facts":
        return bool(re.search(r"peanut|fıstık|izmir|pamuk", a.lower()))
    raise ValueError(check)


def match_rate(check: str, answers: list[str]) -> float:
    return round(100 * sum(persona_score(check, a) for a in answers) / max(1, len(answers)), 1)


def split_prompts(prompts: list[str], seed: int, n_train: int = 45) -> tuple[list[str], list[str]]:
    p = list(prompts)
    random.Random(seed).shuffle(p)
    return p[:n_train], p[n_train:]


def user_messages(persona: Persona, prompt: str) -> list[dict]:
    """How the user asks: just the question. The facts persona told its facts in an earlier
    conversation that is no longer in context, so remembering them has to come from the weights
    (with the facts in context the base model already uses them every time)."""
    return [{"role": "user", "content": prompt}]


# (name, learning rate, NLL weight next to DPO, anchor pairs). Phone cost is about the same.
SETTINGS = [("lr1e-4+anchor", 1e-4, 1.0, 15), ("lr2e-4+anchor", 2e-4, 1.0, 15)]


def anchor_rejected(worked: str) -> str:
    """The same answer without its working: only the final answer line. An anchor pair
    (worked answer > bare answer) differs only in showing work, not in style, so it keeps an
    explicit "think, then answer" request working whatever style the user teaches."""
    lines = [ln for ln in strip_think(worked).strip().splitlines() if ln.strip()]
    return lines[-1].strip() if lines else ""


def correction(persona: Persona, answer: str) -> str:
    """The deterministic part of a user's correction (the model writes the rest)."""
    return answer.lower() if persona.check == "terse" else answer


# ----- the experiment (GPU) ------------------------------------------------------

def _llm(path: str, cfg: Config):
    from vllm import LLM

    return LLM(model=path, max_model_len=4096, gpu_memory_utilization=cfg.gen_gpu_mem, seed=cfg.seed)


def _answer(llm, convs, max_tokens=512, temperature=0.0) -> list[str]:
    from vllm import SamplingParams

    res = llm.chat(convs, SamplingParams(temperature=temperature, max_tokens=max_tokens), use_tqdm=False,
                   chat_template_kwargs={"enable_thinking": False})
    return [strip_think(r.outputs[0].text) for r in res]


def _gsm8k(llm, n: int, seed: int) -> float:
    from candid.evaluate import task_gsm8k

    convs, score = task_gsm8k(n, seed)
    hits = score(_answer(llm, convs, max_tokens=1024))
    return round(100 * sum(hits) / max(1, len(hits)), 1)


def _anchors(llm, n: int, seed: int) -> list[tuple[list[dict], str]]:
    """(conversation, worked answer) for GSM8K train problems the base model gets right."""
    from candid.evaluate import COT, _final, _load, _num

    ds = _load("openai/gsm8k", "main", split="train", n=3 * n, seed=seed)
    convs = [[{"role": "user", "content": ex["question"] + COT}] for ex in ds]
    golds = [float(ex["answer"].split("####")[-1].strip().replace(",", "")) for ex in ds]
    outs = _answer(llm, convs, max_tokens=1024)
    good = [(c, o) for c, o, g in zip(convs, outs, golds)
            if _num(_final(o)) == g and len(o.strip().splitlines()) > 1]
    return good[:n]


def _train(base: str, pairs: list[dict], out: str, cfg: Config, lr: float, nll: float) -> float:
    """Phone-sized LoRA on top of the shipped model; returns training seconds."""
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from trl import DPOConfig, DPOTrainer

    from candid.train import LORA_TARGETS

    processor = AutoProcessor.from_pretrained(base)
    model = AutoModelForImageTextToText.from_pretrained(base, dtype=torch.bfloat16, attn_implementation="sdpa")
    model.config.use_cache = False
    ds = Dataset.from_list(pairs)
    args = DPOConfig(output_dir=out + "-ckpt", per_device_train_batch_size=1, gradient_accumulation_steps=4,
                     learning_rate=lr, num_train_epochs=3, lr_scheduler_type="cosine", warmup_steps=0.1,
                     bf16=True, gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
                     beta=0.1, loss_type=["sigmoid", "sft"], loss_weights=[1.0, nll], max_length=512,
                     truncation_mode="keep_start", save_strategy="no", report_to="none", logging_steps=5,
                     seed=cfg.seed)
    peft = LoraConfig(r=8, lora_alpha=16, target_modules=LORA_TARGETS, task_type="CAUSAL_LM")
    trainer = DPOTrainer(model=model, args=args, train_dataset=ds, processing_class=processor, peft_config=peft)
    t0 = time.time()
    trainer.train()
    secs = time.time() - t0
    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(out, safe_serialization=True)
    processor.save_pretrained(out)
    del trainer, model, merged
    torch.cuda.empty_cache()
    return secs


def _free_gpu() -> None:
    """Call after the caller dropped its last reference to the vLLM engine."""
    import gc

    import torch

    try:
        from vllm.distributed.parallel_state import cleanup_dist_env_and_memory

        cleanup_dist_env_and_memory()
    except Exception:  # noqa: BLE001, S110 - best effort across vLLM versions
        pass
    gc.collect()
    torch.cuda.empty_cache()


def run(cfg: Config, base: str, work: str, n_gsm8k: int = 200) -> dict:
    os.makedirs(work, exist_ok=True)
    results: dict = {}
    llm = _llm(base, cfg)
    before_gsm = _gsm8k(llm, n_gsm8k, cfg.seed)
    plans = {}
    for p in PERSONAS:
        train_p, test_p = split_prompts(p.prompts, cfg.seed)
        rejected = _answer(llm, [user_messages(p, q) for q in train_p])
        sys = [{"role": "system", "content": p.instruction + " " + " ".join(p.facts)}]
        chosen = [correction(p, a) for a in _answer(llm, [sys + user_messages(p, q) for q in train_p], temperature=0.3)]
        before = _answer(llm, [user_messages(p, q) for q in test_p])
        plans[p.name] = (train_p, test_p, chosen, rejected, before)
    anchors = _anchors(llm, max(a for *_, a in SETTINGS), cfg.seed)
    llm = None
    _free_gpu()

    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(base)
    for name, lr, nll, n_anchor in SETTINGS:
        for p in PERSONAS:
            train_p, test_p, chosen, rejected, before = plans[p.name]
            pairs = []
            for q, c, r in zip(train_p, chosen, rejected):
                if c.strip() == r.strip() or not c.strip():
                    continue
                prompt = processor.apply_chat_template(user_messages(p, q), add_generation_prompt=True,
                                                       tokenize=False, enable_thinking=False)
                pairs.append({"prompt": prompt, "chosen": c + "<|im_end|>\n", "rejected": r + "<|im_end|>\n"})
            for conv, worked in anchors[:n_anchor]:
                prompt = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False,
                                                       enable_thinking=False)
                pairs.append({"prompt": prompt, "chosen": worked + "<|im_end|>\n",
                              "rejected": anchor_rejected(worked) + "<|im_end|>\n"})
            random.Random(cfg.seed).shuffle(pairs)
            out = os.path.join(work, f"personal-{p.name}")
            secs = _train(base, pairs, out, cfg, lr, nll)
            llm = _llm(out, cfg)
            after = _answer(llm, [user_messages(p, q) for q in test_p])
            after_gsm = _gsm8k(llm, n_gsm8k, cfg.seed)
            llm = None
            _free_gpu()
            shutil.rmtree(out, ignore_errors=True)
            shutil.rmtree(out + "-ckpt", ignore_errors=True)
            key = f"{p.name}/{name}"
            results[key] = {"pairs": len(pairs), "anchors": min(n_anchor, len(anchors)),
                            "train_seconds": round(secs),
                            "match_before": match_rate(p.check, before), "match_after": match_rate(p.check, after),
                            "gsm8k_before": before_gsm, "gsm8k_after": after_gsm,
                            "sample_after": after[0][:160] if after else ""}
            print("[personal] " + json.dumps({key: results[key]}, ensure_ascii=False)[:480], flush=True)
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    cfg = Config.from_env()
    base = os.environ.get("CANDID_PERSONAL_BASE", f"/workspace/work/runs/{cfg.run}/merged")
    if not os.path.exists(os.path.join(base, "config.json")):
        from huggingface_hub import snapshot_download

        repo = os.environ["CANDID_HUB_REPO"]
        snapshot_download(repo, allow_patterns=[f"runs/{cfg.run}/merged/**"], local_dir="/workspace/hfmodel",
                          token=os.environ.get("HF_TOKEN"))
        base = f"/workspace/hfmodel/runs/{cfg.run}/merged"
    res = run(cfg, base, "/workspace/work/personal")
    with open("/workspace/work/personal/results.json", "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    cfg.store().write_json("personal/results.json", res)  # mirrored to the Hub when a token is set
