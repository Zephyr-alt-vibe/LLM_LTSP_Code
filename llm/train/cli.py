"""Command-line entry point for the RFT pipeline.

Examples
--------

Materialise SFT / RFT JSONLs and step-level rewards (no GPU required)::

    python -m llm_mmi.train prepare \\
        --narrative ukb.all.xlsx \\
        --response  output.xlsx \\
        --outcome   ukb_outcome.xlsx \\
        --work-dir  ./rft_work \\
        --candidate output_run1.xlsx --candidate output_run2.xlsx

Fit Cox only and dump per-record reward / step credit::

    python -m llm_mmi.train score \\
        --narrative ukb.all.xlsx --response output.xlsx \\
        --outcome ukb_outcome.xlsx --out ./rft_work/scores.jsonl

Launch SFT distillation on the materialised JSONL (needs HF/TRL/PEFT)::

    python -m llm_mmi.train distill \\
        --jsonl ./rft_work/sft.jsonl --model meta-llama/Llama-3.1-8B-Instruct

Launch RFT / DPO on the candidate-scored JSONL::

    python -m llm_mmi.train rft \\
        --jsonl ./rft_work/rft_dpo.jsonl --student ./ckpt/sft --method dpo
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import data, regression, step_labels, pipeline, distill, rft


def _build_prepare_parser(sub):
    p = sub.add_parser("prepare", help="materialise SFT/RFT JSONLs and step rewards")
    p.add_argument("--narrative", required=True)
    p.add_argument("--response",  required=True)
    p.add_argument("--outcome",   required=True)
    p.add_argument("--work-dir",  default="./rft_work")
    p.add_argument("--candidate", action="append", default=[],
                   help="additional candidate response files (one per flag)")
    p.add_argument("--step-method", choices=["closed_form", "leave_one_out"],
                   default="closed_form")
    p.add_argument("--rft-method", choices=["rft", "dpo"], default="dpo")
    p.add_argument("--horizon", type=float, default=5.0)
    return p


def _build_score_parser(sub):
    p = sub.add_parser("score", help="fit Cox and dump per-record + per-step rewards")
    p.add_argument("--narrative", required=True)
    p.add_argument("--response",  required=True)
    p.add_argument("--outcome",   required=True)
    p.add_argument("--out",       required=True)
    p.add_argument("--horizon",   type=float, default=5.0)
    return p


def _build_distill_parser(sub):
    p = sub.add_parser("distill", help="run SFT teacher→student distillation")
    p.add_argument("--jsonl", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--output", default="./ckpt/sft")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--lora", action="store_true", default=True)
    p.add_argument("--full-finetune", dest="lora", action="store_false")
    return p


def _build_rft_parser(sub):
    p = sub.add_parser("rft", help="run RFT / DPO on the candidate-scored JSONL")
    p.add_argument("--jsonl", required=True)
    p.add_argument("--student", required=True)
    p.add_argument("--output", default="./ckpt/rft")
    p.add_argument("--method", choices=["rft", "dpo"], default="dpo")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--beta", type=float, default=0.1)
    return p


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m llm_mmi.train",
                                     description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    _build_prepare_parser(sub)
    _build_score_parser(sub)
    _build_distill_parser(sub)
    _build_rft_parser(sub)
    args = parser.parse_args(argv)

    if args.cmd == "prepare":
        cfg = pipeline.PipelineConfig(
            narrative_path=args.narrative,
            response_path=args.response,
            outcome_path=args.outcome,
            work_dir=args.work_dir,
            step_label_method=args.step_method,
            eval_horizon=args.horizon,
            candidate_response_paths=list(args.candidate),
            rft_config=rft.RFTConfig(student_ckpt="",
                                     method=args.rft_method)
                       if args.candidate else None,
        )
        result = pipeline.run(cfg)
        print("done.")
        for k, v in result.items():
            if k != "cox":
                print(f"  {k}: {v}")
        return 0

    if args.cmd == "score":
        records = pipeline.stage_load(pipeline.PipelineConfig(
            narrative_path=args.narrative,
            response_path=args.response,
            outcome_path=args.outcome,
            work_dir=str(Path(args.out).parent),
            eval_horizon=args.horizon,
        ))
        import numpy as np, json
        X = np.stack([r.x_vector for r in records])
        T = np.array([r.time for r in records])
        E = np.array([r.event for r in records], dtype=np.int64)
        cox = regression.fit_cox(X, T, E, data.VARS_USE)
        ll = regression.per_record_partial_loglik(cox, X, T, E)
        credit = step_labels.closed_form_credit(cox, X, T, E,
                                                record_ids=[r.id for r in records])
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            for i, r in enumerate(records):
                f.write(json.dumps({
                    "id": r.id, "reward": float(ll[i]),
                    "step_credit": {step: float(credit.credit[i, k])
                                    for k, step in enumerate(data.VARS_USE)},
                }) + "\n")
        print(f"wrote {len(records)} scored records → {args.out}")
        return 0

    if args.cmd == "distill":
        cfg = distill.DistillConfig(
            model_name_or_path=args.model,
            output_dir=args.output,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            lora=args.lora,
        )
        distill.run_sft(args.jsonl, cfg)
        return 0

    if args.cmd == "rft":
        cfg = rft.RFTConfig(
            student_ckpt=args.student,
            output_dir=args.output,
            method=args.method,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            kl_beta=args.beta,
        )
        rft.run_rft(args.jsonl, cfg)
        return 0

    parser.error("no command")
    return 2


if __name__ == "__main__":
    sys.exit(main())
