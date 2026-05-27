"""Outcome-grounded rejection-sampling fine-tuning (RFT) and DPO.

This module wires together:

1. Candidate generation     — sample ``N`` rollouts from the current
                              student policy at non-zero temperature
                              (or reuse the ``A4.2`` outputs).
2. Reward scoring           — parse each candidate's JSON to a 17-d
                              vector, feed into the fitted Cox model
                              from :mod:`.regression`, score with the
                              partial log-likelihood reward.
3. Step-level credit        — :mod:`.step_labels` decomposition for
                              optional per-step preference pairs.
4. Trainer step             — either keep top-1 per record and re-SFT
                              (RFT/RAFT), or keep (top, bottom) per
                              record and DPO.

As with :mod:`.distill`, the heavy-weight HF/TRL imports happen
inside ``run_*`` so the rest of the package remains importable
without those deps.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence

import numpy as np

from .data import Record, parse_teacher_response, VARS_USE
from .regression import CoxModel, record_reward


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class RFTConfig:
    student_ckpt: str
    output_dir: str = "./ckpt/rft"
    n_candidates: int = 4
    temperature: float = 0.8
    max_new_tokens: int = 1024
    method: str = "dpo"                            # "rft" or "dpo"
    kl_beta: float = 0.1                           # DPO β
    learning_rate: float = 5e-6
    per_device_batch_size: int = 2
    grad_accum_steps: int = 8
    num_epochs: int = 2
    lora: bool = True
    lora_r: int = 16
    margin: float = 0.05                           # min reward gap for DPO pair


# ---------------------------------------------------------------------------
# Reward scoring of a set of candidate responses
# ---------------------------------------------------------------------------
def score_candidates(records: Sequence[Record],
                     candidates: Sequence[Sequence[str]],
                     cox: CoxModel) -> np.ndarray:
    """Score each candidate response with the Cox-grounded reward.

    Parameters
    ----------
    records    : sequence of length n; supplies ``time`` and ``event``.
    candidates : sequence of length n; each element is a list of ``N``
                 candidate response strings for that record.
    cox        : the fitted Cox model.

    Returns
    -------
    rewards : ``(n, N)`` float array; NaN where the candidate failed to
              parse to a complete 17-d vector.
    """
    n = len(records)
    if n == 0:
        return np.zeros((0, 0))
    N = max(len(c) for c in candidates)

    rewards = np.full((n, N), np.nan, dtype=np.float64)

    # Parse all candidates into X, T, E arrays then call record_reward
    # once per candidate slot to amortise the cost of fitting.
    T = np.array([r.time for r in records], dtype=np.float64)
    E = np.array([r.event for r in records], dtype=np.int64)
    for j in range(N):
        Xj = np.full((n, len(VARS_USE)), np.nan)
        for i, cand_list in enumerate(candidates):
            if j < len(cand_list):
                Xj[i], _ = parse_teacher_response(cand_list[i])
        valid = ~np.isnan(Xj).any(axis=1)
        if valid.sum() == 0:
            continue
        rj = record_reward(cox, Xj[valid], T[valid], E[valid])
        rewards[valid, j] = rj
    return rewards


# ---------------------------------------------------------------------------
# Generate candidates from the student
# ---------------------------------------------------------------------------
def generate_candidates(records: Sequence[Record],
                        cfg: RFTConfig,
                        prompt_builder: Callable[[Record], str]) -> List[List[str]]:
    """Sample ``cfg.n_candidates`` responses per record from the student.

    A thin wrapper over ``transformers.pipeline`` / vLLM; switch to
    vLLM in production for throughput.  Returns a nested list:
    ``out[i][j]`` is the j-th candidate response for record i.
    """
    from transformers import (AutoModelForCausalLM, AutoTokenizer,  # type: ignore
                              pipeline)

    tokenizer = AutoTokenizer.from_pretrained(cfg.student_ckpt, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg.student_ckpt, torch_dtype="bfloat16", device_map="auto",
        attn_implementation="flash_attention_2",
    )
    gen = pipeline("text-generation", model=model, tokenizer=tokenizer,
                   max_new_tokens=cfg.max_new_tokens,
                   temperature=cfg.temperature, do_sample=True,
                   num_return_sequences=cfg.n_candidates,
                   pad_token_id=tokenizer.pad_token_id)

    out: List[List[str]] = []
    for rec in records:
        prompt = prompt_builder(rec)
        outs = gen(prompt)
        out.append([o["generated_text"][len(prompt):] for o in outs])
    return out


# ---------------------------------------------------------------------------
# Materialise SFT / DPO training files
# ---------------------------------------------------------------------------
def build_rft_jsonl(records: Sequence[Record],
                    candidates: Sequence[Sequence[str]],
                    rewards: np.ndarray,
                    out_path: str | Path,
                    method: str = "rft",
                    margin: float = 0.05) -> int:
    """Convert candidate responses + rewards into a JSONL training file.

    Layout:

    * ``method == "rft"``  → one record per row with the *highest-reward*
      candidate as the assistant turn (RAFT/best-of-N SFT).
    * ``method == "dpo"``  → one record per row with both the highest
      and lowest reward candidates, when the gap exceeds ``margin``.

    Returns the number of rows written.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows: List[dict] = []
    for i, rec in enumerate(records):
        r = rewards[i]
        if np.all(np.isnan(r)):
            continue
        best = int(np.nanargmax(r))
        worst = int(np.nanargmin(r))
        if method == "rft":
            rows.append({
                "id": rec.id,
                "prompt": rec.narrative,
                "completion": candidates[i][best],
                "reward": float(r[best]),
            })
        elif method == "dpo":
            if best == worst:
                continue
            if (r[best] - r[worst]) < margin:
                continue
            rows.append({
                "id": rec.id,
                "prompt": rec.narrative,
                "chosen": candidates[i][best],
                "rejected": candidates[i][worst],
                "reward_chosen":   float(r[best]),
                "reward_rejected": float(r[worst]),
            })
        else:
            raise ValueError(f"unknown method: {method}")

    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


# ---------------------------------------------------------------------------
# Trainer driver
# ---------------------------------------------------------------------------
def run_rft(jsonl_path: str | Path, cfg: RFTConfig) -> str:
    """Run RFT (best-of-N SFT) or DPO depending on ``cfg.method``.

    Returns the output checkpoint directory.
    """
    from transformers import (AutoModelForCausalLM, AutoTokenizer,  # type: ignore
                              TrainingArguments)
    from datasets import load_dataset                # type: ignore
    ds = load_dataset("json", data_files=str(jsonl_path), split="train")

    tokenizer = AutoTokenizer.from_pretrained(cfg.student_ckpt, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = dict(torch_dtype="bfloat16", device_map="auto",
                        attn_implementation="flash_attention_2")
    model = AutoModelForCausalLM.from_pretrained(cfg.student_ckpt, **model_kwargs)

    peft_config = None
    if cfg.lora:
        from peft import LoraConfig                  # type: ignore
        peft_config = LoraConfig(
            r=cfg.lora_r, lora_alpha=cfg.lora_r * 2,
            lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )

    if cfg.method == "rft":
        from trl import SFTConfig, SFTTrainer        # type: ignore
        sft_cfg = SFTConfig(
            output_dir=cfg.output_dir,
            per_device_train_batch_size=cfg.per_device_batch_size,
            gradient_accumulation_steps=cfg.grad_accum_steps,
            learning_rate=cfg.learning_rate,
            num_train_epochs=cfg.num_epochs,
            bf16=True, logging_steps=20, save_steps=500,
            max_seq_length=4096, packing=True, report_to=[],
        )
        trainer = SFTTrainer(model=model, tokenizer=tokenizer,
                             train_dataset=ds, args=sft_cfg,
                             peft_config=peft_config)
    elif cfg.method == "dpo":
        from trl import DPOConfig, DPOTrainer        # type: ignore
        ref_model = AutoModelForCausalLM.from_pretrained(cfg.student_ckpt,
                                                         **model_kwargs)
        dpo_cfg = DPOConfig(
            output_dir=cfg.output_dir,
            per_device_train_batch_size=cfg.per_device_batch_size,
            gradient_accumulation_steps=cfg.grad_accum_steps,
            learning_rate=cfg.learning_rate,
            num_train_epochs=cfg.num_epochs,
            beta=cfg.kl_beta, bf16=True, logging_steps=20,
            save_steps=500, report_to=[],
        )
        trainer = DPOTrainer(model=model, ref_model=ref_model,
                             tokenizer=tokenizer, train_dataset=ds,
                             args=dpo_cfg, peft_config=peft_config)
    else:
        raise ValueError(f"unknown method: {cfg.method}")

    trainer.train()
    trainer.save_model(cfg.output_dir)
    return cfg.output_dir


__all__ = ["RFTConfig", "score_candidates", "generate_candidates",
           "build_rft_jsonl", "run_rft"]
