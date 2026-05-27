"""Outcome-grounded RFT pipeline for LLM-MMI.

This sub-package contains the *generic* full-flow training pipeline that
extends the frozen-policy inference (Method.md §B.3–B.6) into the
outcome-grounded reverse-credit-assignment RFT regime discussed in
Method.md §D (Scaling to full RLHF).

Module layout
-------------

* :mod:`llm_mmi.train.data`         — narrative + outcome + teacher-CoT
                                       I/O, tokenisation, JSON parsing.
* :mod:`llm_mmi.train.regression`   — Cox PH fit + IPCW Brier + per-record
                                       partial-likelihood contributions.
* :mod:`llm_mmi.train.step_labels`  — reverse step-level credit assignment
                                       (closed-form Cox decomposition +
                                       leave-one-step-out ablation).
* :mod:`llm_mmi.train.distill`      — SFT teacher→student distillation
                                       skeleton (HF Transformers + PEFT).
* :mod:`llm_mmi.train.rft`          — outcome-grounded rejection-sampling
                                       fine-tuning / DPO skeleton.
* :mod:`llm_mmi.train.pipeline`     — end-to-end orchestrator that wires
                                       the above into one ``run()`` call.
* :mod:`llm_mmi.train.cli`          — ``python -m llm_mmi.train`` CLI.

The interfaces are dataset-agnostic; LLM-MMI is the reference adapter.
"""

from . import data, regression, step_labels, distill, rft, pipeline  # noqa: F401

__all__ = ["data", "regression", "step_labels", "distill", "rft", "pipeline"]
