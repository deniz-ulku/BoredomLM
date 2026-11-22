"""Preference training: DPO + a small NLL term on the chosen answer (RPO-style).

* LoRA on the language model only (attention, Gated DeltaNet projections, MLP).
  The vision tower and the DeltaNet gates stay frozen, so image understanding and
  long-context behaviour are not disturbed.
* DPO's KL anchor to the frozen base (the reference is the same weights with the
  adapter switched off) is the second guard against losing capability.
* Resumable: a checkpoint is saved and uploaded every cfg.ckpt_minutes, and a new
  machine continues from the latest one.
"""

from __future__ import annotations

import glob
import logging
import os
import time

from candid.config import Config

log = logging.getLogger("candid.train")

LORA_TARGETS = (
    r"model\.language_model\.layers\.\d+\."
    r"(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)"
    r"|linear_attn\.(?:in_proj_qkv|in_proj_z|out_proj)"
    r"|mlp\.(?:gate_proj|up_proj|down_proj))"
)


def latest_checkpoint(ckpt_dir: str) -> str | None:
    paths = glob.glob(os.path.join(ckpt_dir, "checkpoint-*"))
    paths = [p for p in paths if os.path.exists(os.path.join(p, "trainer_state.json"))]
    return max(paths, key=lambda p: int(p.rsplit("-", 1)[-1]), default=None)


def cap_image_size(processor, max_pixels: int) -> None:
    ip = getattr(processor, "image_processor", None)
    if ip is not None:
        ip.size = {**dict(ip.size), "longest_edge": max_pixels}


def drop_too_long(ds, processor, max_length: int):
    """Drop pairs that would not fit: truncation would cut image tokens and TRL raises."""
    def fits(ex):
        images = ex["images"] or None
        for side in ("chosen", "rejected"):
            ids = processor(text=[ex["prompt"] + ex[side]], images=images)["input_ids"][0]
            if len(ids) > max_length:
                return False
        return True

    before = len(ds)
    ds = ds.filter(fits)
    log.info("dropped %d of %d pairs longer than %d tokens", before - len(ds), before, max_length)
    return ds


def _callbacks(cfg: Config, store):
    from transformers import TrainerCallback

    class TimedSave(TrainerCallback):
        """Save on a wall-clock interval instead of a step count."""

        def __init__(self):
            self.last = time.time()

        def on_step_end(self, args, state, control, **kw):
            if time.time() - self.last >= cfg.ckpt_minutes * 60:
                control.should_save = True
                self.last = time.time()

    class Upload(TrainerCallback):
        def on_save(self, args, state, control, **kw):
            if latest_checkpoint(store.local("ckpt")):
                store.push("ckpt", message=f"checkpoint step {state.global_step}", replace=True)
                store.update_state(progress=f"train step {state.global_step}/{state.max_steps}")

    return [TimedSave(), Upload()]


def run(cfg: Config, store) -> dict:
    import torch
    from datasets import load_from_disk
    from peft import LoraConfig
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from trl import DPOConfig, DPOTrainer

    store.pull("pairs")
    ds = load_from_disk(store.local("pairs"))
    log.info("pairs: %d", len(ds))
    store.pull("ckpt")
    resume = latest_checkpoint(store.local("ckpt"))
    log.info("resume from: %s", resume)

    processor = AutoProcessor.from_pretrained(cfg.base_model)
    cap_image_size(processor, cfg.train_max_pixels)
    ds = drop_too_long(ds, processor, cfg.max_length)
    model = AutoModelForImageTextToText.from_pretrained(cfg.base_model, dtype=torch.bfloat16,
                                                        attn_implementation="sdpa")
    model.config.use_cache = False

    peft_config = LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
                             target_modules=LORA_TARGETS, task_type="CAUSAL_LM")

    args = DPOConfig(
        output_dir=store.local("ckpt"),
        per_device_train_batch_size=1,       # mixed image / text-only examples: keep one per batch
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=0.05,                   # < 1 is read as a fraction of total steps
        num_train_epochs=cfg.epochs,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        beta=cfg.dpo_beta,
        loss_type=["sigmoid", "sft"],
        loss_weights=[1.0, cfg.sft_weight],
        max_length=cfg.max_length,
        truncation_mode="keep_start",
        remove_unused_columns=False,
        logging_steps=5,
        save_strategy="no",                  # saving is driven by TimedSave
        save_total_limit=1,
        report_to="none",
        seed=cfg.seed,
        dataloader_num_workers=2,
    )

    trainer = DPOTrainer(model=model, args=args, train_dataset=ds, processing_class=processor,
                         peft_config=peft_config, callbacks=_callbacks(cfg, store))
    trainer.model.print_trainable_parameters()
    result = trainer.train(resume_from_checkpoint=resume)

    trainer.model.save_pretrained(store.local("adapter"))
    AutoProcessor.from_pretrained(cfg.base_model).save_pretrained(store.local("adapter"))  # original image size
    store.push("adapter", replace=True)

    logs = [h for h in trainer.state.log_history if "loss" in h]
    out = {"train_loss": result.training_loss, "steps": trainer.state.global_step, "last_logs": logs[-5:]}
    store.write_json("train_stats.json", out)
    log.info("training done: %s", out)
    return out


def merge(cfg: Config, store) -> str:
    """Base + adapter -> merged bf16 checkpoint (local only; recreated on a new machine when needed)."""
    out = store.local("merged")
    if os.path.exists(os.path.join(out, "config.json")):
        return out
    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    if not store.pull("adapter"):
        raise RuntimeError("no adapter to merge")
    base = AutoModelForImageTextToText.from_pretrained(cfg.base_model, dtype=torch.bfloat16)
    merged = PeftModel.from_pretrained(base, store.local("adapter")).merge_and_unload()
    merged.config.use_cache = True
    merged.save_pretrained(out, safe_serialization=True)
    AutoProcessor.from_pretrained(cfg.base_model).save_pretrained(out)
    return out
