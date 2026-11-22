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
    run: str = "v1"               # run name (remote/RUN picks it on the machines)
    work_dir: str = "hub/runs/v1"  # local mirror of runs/<run>/ (see candid/store.py)
    seed: int = 0

    # Prompt mix (number of prompts drawn per bucket; see candid/sources.py).
    # v0 lesson: the base refused only ~8% of OR-Bench-80k prompts, so v1 screens many of them
    # cheaply and builds pairs only from the ones it actually refuses. General and image prompts
    # become pairs only when the default answer breaks a style rule.
    n_general_en: int = 1500
    n_general_tr: int = 1500
    n_over_refusal: int = 15000   # screened; ~8% are expected to be refused and become pairs
    n_hard_limit: int = 0         # base refuses these on its own; no training either way
    n_image: int = 1500
    n_image_pref: int = 800       # off-policy hallucination preference pairs (RLAIF-V)

    # On-policy generation (vLLM). Work is split into shards that are saved as they finish.
    shard_size: int = 500
    screen_max_tokens: int = 96   # enough to see whether an answer starts with a refusal
    gen_samples: int = 4          # compliant candidates per refused prompt
    gen_temperature: float = 0.7
    gen_max_tokens: int = 1536
    gen_max_model_len: int = 8192
    gen_gpu_mem: float = 0.85
    min_edit_ratio: float = 0.6   # a rewrite shorter than this share of the original lost content

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

    # Evaluation gate: a capability benchmark fails when the drop exceeds both this many points
    # and the 95% noise band of the difference (two independent samples of eval_limit questions).
    max_capability_drop: float = 1.0
    eval_limit: int = 1000        # examples per capability benchmark
    eval_behaviour_limit: int = 500  # over-refusal prompts

    # Outputs.
    hub_repo: str = ""            # e.g. "your-name/candid-4b-v0" (private). Empty = no upload.
    gguf_quants: list[str] = field(default_factory=lambda: ["Q4_K_M", "Q8_0"])
    # Stages to run again once for a finished run (e.g. "export" after a converter change).
    # Applied once per value: state["redo_applied"] remembers it, so restarts do not loop;
    # extra tags that are not stage names ("export,2") make a new value for another redo.
    redo: list[str] = field(default_factory=list)

    @classmethod
    def from_env(cls) -> Config:
        cfg = cls()
        for f in fields(cls):
            raw = os.environ.get(f"CANDID_{f.name.upper()}")
            if raw is None:
                continue
            kind = str(f.type)  # declared type, not the default's runtime type (15 vs 15.0)
            if kind == "bool":
                val = raw.lower() in ("1", "true", "yes")
            elif kind == "int":
                val = int(raw)
            elif kind == "float":
                val = float(raw)
            elif kind.startswith("list"):
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
