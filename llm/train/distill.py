"""SFT teacher → student distillation.

Implements the *first* training stage of the RFT pipeline:

    L_SFT(θ) = -E_{(n, y) ~ D_teacher} log π_θ(y | n)

where ``y`` is the full ``Response`` string emitted by the frozen
policy (JSON + per-task CoT).  The student is typically a 7B–13B base
checkpoint trained with LoRA / QLoRA so the entire stage fits on the
2×80 GB ceiling discussed in Method.md §D.

This module is a *runnable skeleton* over HuggingFace Transformers +
TRL ``SFTTrainer`` — the heavy lifting (FlashAttention, ZeRO, sequence
packing) is delegated to TRL.  It is intentionally thin so it can be
swapped for axolotl, llama-factory, or unsloth without touching the
data-loading or step-labelling layers.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional

from .data import (LLMMMIExcelAdapter, Record, stream_chat_examples,
                   VARS_USE)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class DistillConfig:
    model_name_or_path: str = "meta-llama/Llama-3.1-8B-Instruct"
    output_dir: str = "./ckpt/sft"
    max_seq_len: int = 4096
    learning_rate: float = 2e-5
    per_device_batch_size: int = 4
    grad_accum_steps: int = 4
    num_epochs: int = 3
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    bf16: bool = True
    gradient_checkpointing: bool = True
    lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    system_prompt: Optional[str] = None
    # Sample-weighting hook: if a per-record weight file is supplied (the
    # output of step_labels.credits_to_step_rewards aggregated to the
    # record level), the trainer up-weights high-reward records.
    sample_weight_path: Optional[str] = None


# ---------------------------------------------------------------------------
# Dataset materialisation
# ---------------------------------------------------------------------------
def write_chat_jsonl(adapter: Iterable[Record],
                     out_path: str | Path,
                     system_prompt: Optional[str] = None,
                     require_teacher: bool = True) -> int:
    """Stream chat-format examples to a JSONL file usable by TRL/axolotl.

    Returns the number of examples written.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w", encoding="utf-8") as f:
        for ex in stream_chat_examples(adapter,
                                       system_prompt=system_prompt,
                                       require_teacher=require_teacher):
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
            n += 1
    return n


# ---------------------------------------------------------------------------
# Trainer driver
# ---------------------------------------------------------------------------
def run_sft(jsonl_path: str | Path, cfg: DistillConfig) -> str:
    """Launch SFT distillation.  Returns the output checkpoint dir.

    Heavy deps (``transformers``, ``trl``, ``peft``) are imported lazily
    so the rest of the package stays importable in pure-NumPy
    environments (data parsing, regression, step labels).
    """
    from transformers import (AutoModelForCausalLM, AutoTokenizer,  # type: ignore
                              TrainingArguments)
    from datasets import load_dataset                # type: ignore
    from trl import SFTConfig, SFTTrainer            # type: ignore

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path,
                                              use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds = load_dataset("json", data_files=str(jsonl_path), split="train")

    model_kwargs = dict(torch_dtype="bfloat16" if cfg.bf16 else "float16",
                        device_map="auto",
                        attn_implementation="flash_attention_2")
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name_or_path,
                                                 **model_kwargs)
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    peft_config = None
    if cfg.lora:
        from peft import LoraConfig                  # type: ignore
        peft_config = LoraConfig(
            r=cfg.lora_r, lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=cfg.target_modules,
            bias="none", task_type="CAUSAL_LM",
        )

    sft_cfg = SFTConfig(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=cfg.grad_accum_steps,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_epochs,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        bf16=cfg.bf16, logging_steps=20, save_steps=500,
        max_seq_length=cfg.max_seq_len,
        packing=True,
        gradient_checkpointing=cfg.gradient_checkpointing,
        report_to=[],
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds,
        args=sft_cfg,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(cfg.output_dir)
    return cfg.output_dir


__all__ = ["DistillConfig", "write_chat_jsonl", "run_sft"]
