"""Every knob of the v0 pipeline in one place.

Values can be overridden from the environment (CANDID_<FIELD>=...), which is how
the vast.ai launcher passes run-specific settings to the remote machine.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields


@dataclass
class Config:
    # Base model: Apache-2.0, natively multimodal, 3:1 Gated DeltaNet / attention hybrid.
    base_model: str = "Qwen/Qwen3.5-4B"
    run: str = "v0"               # run name; state lives under runs/<run>/ in the Hub repo
    work_dir: str = "hub/runs/v0"  # local mirror of runs/<run>/ (see candid/store.py)
    seed: int = 0

    # Prompt mix (number of prompts drawn per bucket; see candid/sources.py). About half the
    # prompts become pairs (smoke run: 31 of 57), so this is ~5-6k pairs, ~4 h of training on
    # an RTX 5090. Over-refusal stays largest: the base refused 80% of OR-Bench-Hard prompts.
    n_general_en: int = 2000
    n_general_tr: int = 2000
    n_over_refusal: int = 2500
    n_hard_limit: int = 300
    n_image: int = 2000
    n_image_pref: int = 1000  # off-policy hallucination preference pairs (RLAIF-V)

    # On-policy generation (vLLM). Work is split into shards that are saved as they finish.
    shard_size: int = 500
    gen_samples: int = 3          # best-of-n samples with the behaviour spec
    gen_temperature: float = 0.7
    gen_max_tokens: int = 1536
    gen_max_model_len: int = 8192
    gen_gpu_mem: float = 0.85
    use_self_judge: bool = True   # base model compares default vs spec answer, both orders

    # Preference training (LoRA on the language model only; the vision tower stays frozen).
    lora_r: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    learning_rate: float = 5e-6
    dpo_beta: float = 0.1
    sft_weight: float = 0.2       # RPO-style NLL on the chosen answer, keeps answers fluent
    epochs: float = 1.0
    grad_accum: int = 16
    max_length: int = 2048
    # Image size during training only (vision tower is frozen; this bounds image tokens so
    # prompt + answer fit in max_length). 1024*1024 px is ~1000 tokens for Qwen3.5.
    train_max_pixels: int = 1024 * 1024
    ckpt_minutes: float = 15      # save + upload a training checkpoint this often

    # Evaluation gate: tuned model may lose at most this many points on any capability benchmark.
    max_capability_drop: float = 1.0
    eval_limit: int = 300         # examples per benchmark

    # Outputs.
    hub_repo: str = ""            # e.g. "your-name/candid-4b-v0" (private). Empty = no upload.
    gguf_quants: list[str] = field(default_factory=lambda: ["Q4_K_M", "Q8_0"])

    @classmethod
    def from_env(cls) -> Config:
        cfg = cls()
        for f in fields(cls):
            raw = os.environ.get(f"CANDID_{f.name.upper()}")
            if raw is None:
                continue
            cur = getattr(cfg, f.name)
            if isinstance(cur, bool):
                val = raw.lower() in ("1", "true", "yes")
            elif isinstance(cur, int):
                val = int(raw)
            elif isinstance(cur, float):
                val = float(raw)
            elif isinstance(cur, list):
                val = [x for x in raw.split(",") if x]
            else:
                val = raw
            setattr(cfg, f.name, val)
        if "CANDID_WORK_DIR" not in os.environ:
            cfg.work_dir = os.path.join("hub", "runs", cfg.run)
        return cfg

    def store(self):
        from candid.store import Store

        return Store(self.hub_repo or None, self.run, self.work_dir)

    def path(self, *parts: str) -> str:
        p = os.path.join(self.work_dir, *parts)
        os.makedirs(os.path.dirname(p) if os.path.splitext(p)[1] else p, exist_ok=True)
        return p
