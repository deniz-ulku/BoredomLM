"""Preference training: DPO + a small NLL term on the chosen answer (RPO-style).

* LoRA on the language model only (attention, Gated DeltaNet projections, MLP).
  The vision tower and the DeltaNet gates stay frozen, so image understanding and
  long-context behaviour are not disturbed.
* DPO's KL anchor to the frozen base (the reference is the same weights with the
  adapter switched off) is the second guard against losing capability.
* Output: LoRA adapter + a merged bf16 checkpoint for evaluation and export.

Usage (on the GPU machine):  python -m candid.train
"""

from __future__ import annotations

import json
import logging

from candid.config import Config

log = logging.getLogger("candid.train")

LORA_TARGETS = (
    r"model\.language_model\.layers\.\d+\."
    r"(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)"
    r"|linear_attn\.(?:in_proj_qkv|in_proj_z|out_proj)"
    r"|mlp\.(?:gate_proj|up_proj|down_proj))"
)


def run(cfg: Config) -> dict:
    import torch
    from datasets import load_from_disk
    from peft import LoraConfig
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from trl import DPOConfig, DPOTrainer

    ds = load_from_disk(cfg.path("pairs"))
    log.info("pairs: %d", len(ds))

    processor = AutoProcessor.from_pretrained(cfg.base_model)
    model = AutoModelForImageTextToText.from_pretrained(cfg.base_model, dtype=torch.bfloat16,
                                                        attn_implementation="sdpa")
    model.config.use_cache = False

    peft_config = LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
                             target_modules=LORA_TARGETS, task_type="CAUSAL_LM")

    args = DPOConfig(
        output_dir=cfg.path("ckpt"),
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
        logging_steps=10,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=1,
        report_to="none",
        seed=cfg.seed,
        dataloader_num_workers=2,
    )

    trainer = DPOTrainer(model=model, args=args, train_dataset=ds, processing_class=processor,
                         peft_config=peft_config)
    trainer.model.print_trainable_parameters()
    result = trainer.train()

    trainer.model.save_pretrained(cfg.path("adapter"))
    processor.save_pretrained(cfg.path("adapter"))

    merged = trainer.model.merge_and_unload()
    merged.config.use_cache = True
    merged.save_pretrained(cfg.path("merged"), safe_serialization=True)
    processor.save_pretrained(cfg.path("merged"))

    logs = [h for h in trainer.state.log_history if "loss" in h]
    out = {"train_loss": result.training_loss, "steps": trainer.state.global_step, "last_logs": logs[-5:]}
    with open(cfg.path("train_stats.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)
    log.info("training done: %s", out)
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    run(Config.from_env())
