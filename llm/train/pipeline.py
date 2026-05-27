"""End-to-end orchestrator for the outcome-grounded RFT pipeline.

Wires the five modules together:

    1. data        — load narratives + teacher CoT + outcomes.
    2. regression  — fit Cox PH on the teacher's 17-d output.
    3. step_labels — reverse step-level credit assignment per record.
    4. distill     — SFT teacher → student.
    5. rft         — best-of-N + DPO refinement using the Cox reward.

A single ``run(cfg)`` call produces a ready-to-train RFT JSONL plus
the step-level reward file, which can be consumed by the student
trainer or exported for offline RM training.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

from .data import (LLMMMIExcelAdapter, Record, VARS_USE,
                   stream_chat_examples)
from .regression import CoxModel, fit_cox, harrell_c_index, ipcw_brier
from .step_labels import (closed_form_credit, leave_one_step_out_credit,
                          credits_to_step_rewards)
from .distill import write_chat_jsonl, DistillConfig
from .rft import score_candidates, build_rft_jsonl, RFTConfig


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------
@dataclass
class PipelineConfig:
    narrative_path: str
    response_path:  str
    outcome_path:   str
    work_dir:       str = "./rft_work"

    cox_l2:         float = 1e-4
    eval_horizon:   float = 5.0

    step_label_method: str = "closed_form"    # or "leave_one_out"
    step_label_normalise: str = "zscore"

    sft_config: DistillConfig = field(default_factory=DistillConfig)
    rft_config: Optional[RFTConfig] = None    # set when running RFT step

    candidate_response_paths: List[str] = field(default_factory=list)
    """If a list of ``output_run{k}.xlsx`` files is supplied (e.g. from
    README §7 stability runs), they are scored as additional candidates
    against the teacher's primary response and used to build the
    DPO pairs file."""


# ---------------------------------------------------------------------------
# Stage runners
# ---------------------------------------------------------------------------
def stage_load(cfg: PipelineConfig) -> List[Record]:
    """Stage 1 — materialise records from the LLM-MMI adapter.

    For 1M-row corpora this fits in RAM (~2 GB).  For 10M, prefer the
    streaming iterator ``LLMMMIExcelAdapter(...)`` directly and chunk
    the downstream stages — Cox fitting on a uniform subsample is
    statistically sufficient.
    """
    adapter = LLMMMIExcelAdapter(cfg.narrative_path, cfg.response_path,
                                 cfg.outcome_path)
    records = [r for r in adapter
               if r.has_x() and r.has_outcome()]
    print(f"[load] complete records: {len(records)}")
    return records


def stage_fit_cox(records: List[Record], cfg: PipelineConfig) -> CoxModel:
    """Stage 2 — fit the outcome regression on the teacher's outputs."""
    X = np.stack([r.x_vector for r in records])
    T = np.array([r.time for r in records], dtype=np.float64)
    E = np.array([r.event for r in records], dtype=np.int64)
    cox = fit_cox(X, T, E, feature_names=VARS_USE, l2=cfg.cox_l2)
    # Cheap summary metrics on a 5k subsample for sanity.
    if len(records) > 5000:
        idx = np.random.default_rng(0).choice(len(records), 5000, replace=False)
        Xs, Ts, Es = X[idx], T[idx], E[idx]
    else:
        Xs, Ts, Es = X, T, E
    c = harrell_c_index(cox, Xs, Ts, Es)
    bs = ipcw_brier(cox, Xs, Ts, Es, horizon=cfg.eval_horizon)
    print(f"[cox] C={c:.4f}  IPCW-Brier@{cfg.eval_horizon}={bs:.4f}")
    return cox


def stage_step_labels(records: List[Record], cox: CoxModel,
                      cfg: PipelineConfig) -> dict:
    """Stage 3 — reverse step-level credit assignment."""
    X = np.stack([r.x_vector for r in records])
    T = np.array([r.time for r in records], dtype=np.float64)
    E = np.array([r.event for r in records], dtype=np.int64)
    ids = [r.id for r in records]
    if cfg.step_label_method == "closed_form":
        credit = closed_form_credit(cox, X, T, E, record_ids=ids)
    elif cfg.step_label_method == "leave_one_out":
        credit = leave_one_step_out_credit(cox, X, T, E, record_ids=ids)
    else:
        raise ValueError(f"unknown step_label_method: {cfg.step_label_method}")
    rewards = credits_to_step_rewards(credit, normalise=cfg.step_label_normalise)

    out_path = Path(cfg.work_dir) / "step_rewards.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"method": rewards["method"],
                            "normalise": rewards["normalise"]}) + "\n")
        for row in rewards["rewards"]:
            f.write(json.dumps(row) + "\n")
    print(f"[step] wrote {len(rewards['rewards'])} step rewards → {out_path}")
    return rewards


def stage_sft_jsonl(records: List[Record], cfg: PipelineConfig) -> Path:
    """Stage 4a — emit the SFT distillation JSONL."""
    out_path = Path(cfg.work_dir) / "sft.jsonl"
    n = write_chat_jsonl(records, out_path,
                         system_prompt=cfg.sft_config.system_prompt,
                         require_teacher=True)
    print(f"[sft] wrote {n} chat examples → {out_path}")
    return out_path


def stage_rft_jsonl(records: List[Record], cox: CoxModel,
                    cfg: PipelineConfig) -> Optional[Path]:
    """Stage 4b — score candidate runs and emit RFT / DPO JSONL.

    Requires :attr:`PipelineConfig.candidate_response_paths` to be set;
    if empty, the function returns ``None`` and the caller falls back
    to pure SFT distillation.
    """
    if not cfg.candidate_response_paths or cfg.rft_config is None:
        print("[rft] no candidate responses supplied — skipping.")
        return None

    # Load the additional candidate response files keyed by record id.
    import pandas as pd
    cand_per_record: List[List[str]] = [[] for _ in records]
    id_to_index = {r.id: i for i, r in enumerate(records)}
    for path in cfg.candidate_response_paths:
        df = pd.read_excel(path) if path.endswith((".xlsx", ".xls")) \
             else pd.read_parquet(path)
        for _, row in df.iterrows():
            idx = id_to_index.get(str(row["ID"]))
            if idx is not None:
                cand_per_record[idx].append(str(row["Response"]))
    # Include the primary teacher response as candidate 0.
    for i, rec in enumerate(records):
        if rec.teacher_response:
            cand_per_record[i].insert(0, rec.teacher_response)

    rewards = score_candidates(records, cand_per_record, cox)

    out_path = Path(cfg.work_dir) / f"rft_{cfg.rft_config.method}.jsonl"
    n = build_rft_jsonl(records, cand_per_record, rewards, out_path,
                        method=cfg.rft_config.method,
                        margin=cfg.rft_config.margin)
    print(f"[rft] wrote {n} {cfg.rft_config.method.upper()} rows → {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Top-level entry
# ---------------------------------------------------------------------------
def run(cfg: PipelineConfig) -> dict:
    """Run all stages up to RFT JSONL emission.

    The two compute-heavy stages — actual SFT / DPO training — are *not*
    invoked here automatically; callers should pass the returned
    ``sft_jsonl`` / ``rft_jsonl`` paths into :func:`distill.run_sft` /
    :func:`rft.run_rft` from a GPU node.
    """
    Path(cfg.work_dir).mkdir(parents=True, exist_ok=True)
    records = stage_load(cfg)
    cox = stage_fit_cox(records, cfg)
    step_rewards = stage_step_labels(records, cox, cfg)
    sft_path = stage_sft_jsonl(records, cfg)
    rft_path = stage_rft_jsonl(records, cox, cfg)
    return {
        "n_records":   len(records),
        "cox":         cox,
        "sft_jsonl":   str(sft_path),
        "rft_jsonl":   str(rft_path) if rft_path else None,
        "step_rewards_n": len(step_rewards["rewards"]),
    }


__all__ = ["PipelineConfig", "run",
           "stage_load", "stage_fit_cox", "stage_step_labels",
           "stage_sft_jsonl", "stage_rft_jsonl"]
