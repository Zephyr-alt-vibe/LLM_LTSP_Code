"""Reverse step-level credit assignment.

A teacher response consists of a sequence of *steps*

    step_k = (rationale_k, value_k)

where ``rationale_k`` is the ``"inference process N"`` text and
``value_k`` is the corresponding numeric / ordinal field (e.g.
``"cardiovascular system age": 65``).  We want a *per-(record, step)*
reward signal that can be used as:

* an SFT weight (sample importance),
* a reward-model regression target, or
* a DPO preference label when two candidate steps are compared.

Two complementary estimators are implemented:

1. :func:`closed_form_credit`
   Decomposes the Cox linear predictor

        η_i = Σ_k β_k · z_{i,k},   z_{i,k} = (x_{i,k} − μ_k) / σ_k

   into per-step contributions ``c_{i,k} = β_k · z_{i,k}``.  This is
   the *exact* attribution of η to the steps that produced each
   ``x_{i,k}`` and requires no extra LLM calls.  Sign of ``c_{i,k}``
   relative to the observed event direction (event = 1 ⇒ we want
   η_i to be *large*) gives a binary preference label.

2. :func:`leave_one_step_out_credit`
   Re-evaluates the partial log-likelihood with one step's value
   replaced by its cohort mean (a "step ablation").  The drop in
   ``ℓ_i`` quantifies that step's *empirical* marginal value beyond
   the closed-form decomposition — useful when downstream
   transformations (Level-2 recalibration, non-linear aggregation)
   are present.

The output of both estimators is a 2-D array ``C ∈ R^{n × p}`` where
``p = len(VARS_USE)``.  Helper functions convert ``C`` into:

* :func:`credits_to_step_rewards` — per-step scalar reward;
* :func:`credits_to_preference_pairs` — DPO-style (chosen, rejected)
  step pairs for the same record but different candidate values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import numpy as np

from .data import VARS_USE
from .regression import CoxModel, per_record_partial_loglik


# ---------------------------------------------------------------------------
# Container
# ---------------------------------------------------------------------------
@dataclass
class StepCredit:
    """Per-(record, step) credit matrix and its provenance."""

    record_ids: Sequence[str]
    step_names: Sequence[str]
    credit: np.ndarray              # (n, p), float64
    method: str                     # "closed_form" or "leave_one_out"

    def step_rewards(self) -> np.ndarray:
        """Reward sign aligned with the outcome direction is the raw
        credit value; users that want zero-mean rewards should subtract
        ``self.credit.mean(axis=0)``."""
        return self.credit


# ---------------------------------------------------------------------------
# Estimator 1: closed-form Cox decomposition
# ---------------------------------------------------------------------------
def closed_form_credit(model: CoxModel,
                       X: np.ndarray,
                       T: np.ndarray,
                       E: np.ndarray,
                       record_ids: Optional[Sequence[str]] = None) -> StepCredit:
    """Closed-form per-step contribution to η_i, signed by outcome.

    For each record ``i`` and step ``k`` we compute

        c_{i,k} = sign(direction_i) · β_k · (x_{i,k} − μ_k) / σ_k,

    where ``direction_i = +1`` if E_i = 1 (event) and ``-1`` otherwise.
    A positive credit means the step *agreed* with the observed
    outcome direction; a negative credit means the step pushed η
    against the outcome.

    Records with NaN features get NaN credits in those columns; callers
    should filter or impute upstream.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[None, :]
    z = (X - model.mean) / model.std
    contrib = z * model.beta                        # (n, p)
    direction = np.where(E == 1, 1.0, -1.0)         # (n,)
    credit = contrib * direction[:, None]
    if record_ids is None:
        record_ids = [str(i) for i in range(len(X))]
    return StepCredit(record_ids=list(record_ids),
                      step_names=list(model.feature_names),
                      credit=credit,
                      method="closed_form")


# ---------------------------------------------------------------------------
# Estimator 2: leave-one-step-out ablation
# ---------------------------------------------------------------------------
def leave_one_step_out_credit(model: CoxModel,
                              X: np.ndarray, T: np.ndarray, E: np.ndarray,
                              record_ids: Optional[Sequence[str]] = None,
                              impute: str = "mean") -> StepCredit:
    """For each step k, replace column k with its impute value and
    recompute the per-record partial log-likelihood.  The drop

        Δℓ_{i,k} = ℓ_i(full) − ℓ_i(without step k)

    is the marginal contribution of step k for record i; positive ⇒
    step k *helped* explain the event.

    ``impute = "mean"`` (cohort mean) or ``"median"`` are supported.
    Cost: O(p) Cox-likelihood evaluations, each O(n log n).
    """
    X = np.asarray(X, dtype=np.float64)
    n, p = X.shape

    if impute == "median":
        replace_val = np.nanmedian(X, axis=0)
    else:
        replace_val = np.nanmean(X, axis=0)

    full = per_record_partial_loglik(model, X, T, E)
    credit = np.zeros((n, p), dtype=np.float64)
    for k in range(p):
        X_ablated = X.copy()
        X_ablated[:, k] = replace_val[k]
        ll_k = per_record_partial_loglik(model, X_ablated, T, E)
        credit[:, k] = full - ll_k

    if record_ids is None:
        record_ids = [str(i) for i in range(n)]
    return StepCredit(record_ids=list(record_ids),
                      step_names=list(model.feature_names),
                      credit=credit,
                      method="leave_one_out")


# ---------------------------------------------------------------------------
# Conversion to RFT / DPO targets
# ---------------------------------------------------------------------------
def credits_to_step_rewards(credit: StepCredit,
                            normalise: str = "zscore") -> dict:
    """Convert a credit matrix into a flat list of (record_id, step,
    reward) triples suitable for an SFT/RM data file.

    ``normalise`` ∈ {"none", "zscore", "rank"}.
    """
    C = credit.credit
    if normalise == "zscore":
        mu = np.nanmean(C, axis=0)
        sd = np.nanstd(C, axis=0)
        sd = np.where(sd > 0, sd, 1.0)
        C = (C - mu) / sd
    elif normalise == "rank":
        ranks = np.full_like(C, np.nan, dtype=np.float64)
        for k in range(C.shape[1]):
            col = C[:, k]
            valid = ~np.isnan(col)
            if valid.sum() == 0:
                continue
            ranks[valid, k] = np.argsort(np.argsort(col[valid])) / max(valid.sum() - 1, 1)
        C = 2 * ranks - 1                           # → [-1, 1]
    elif normalise != "none":
        raise ValueError(f"unknown normalise mode: {normalise}")

    out: List[dict] = []
    for i, rid in enumerate(credit.record_ids):
        for k, step in enumerate(credit.step_names):
            v = C[i, k]
            if not np.isnan(v):
                out.append({"id": rid, "step": step, "reward": float(v)})
    return {"method": credit.method,
            "normalise": normalise,
            "rewards": out}


def credits_to_preference_pairs(credit_a: StepCredit,
                                credit_b: StepCredit,
                                margin: float = 0.05) -> List[dict]:
    """Build DPO-style step-level preference pairs from two candidate
    runs.

    For each (record, step) where both candidates have a finite credit
    and their absolute difference exceeds ``margin``, emit

        {"id": …, "step": step,
         "chosen":   candidate_with_higher_credit,
         "rejected": candidate_with_lower_credit,
         "margin": Δ}

    where ``candidate_…`` is either ``"A"`` or ``"B"``.  These records
    can be joined back to the corresponding rationale + value strings
    upstream to materialise full DPO training pairs.
    """
    if credit_a.step_names != credit_b.step_names:
        raise ValueError("step_names mismatch")
    if list(credit_a.record_ids) != list(credit_b.record_ids):
        raise ValueError("record_ids mismatch")

    pairs: List[dict] = []
    Ca, Cb = credit_a.credit, credit_b.credit
    for i, rid in enumerate(credit_a.record_ids):
        for k, step in enumerate(credit_a.step_names):
            a, b = Ca[i, k], Cb[i, k]
            if np.isnan(a) or np.isnan(b):
                continue
            diff = a - b
            if abs(diff) < margin:
                continue
            pairs.append({
                "id": rid, "step": step,
                "chosen":   "A" if diff > 0 else "B",
                "rejected": "B" if diff > 0 else "A",
                "margin": float(abs(diff)),
            })
    return pairs


# ---------------------------------------------------------------------------
# Convenience: process candidate samples produced by the inference run
# ---------------------------------------------------------------------------
def credits_from_candidates(model: CoxModel,
                            candidate_X: Iterable[np.ndarray],
                            T: np.ndarray,
                            E: np.ndarray,
                            record_ids: Sequence[str],
                            method: str = "closed_form") -> List[StepCredit]:
    """Score a list of candidate matrices (e.g. the N samples produced
    by ``LLM_MMI_PRESET=A4.2``) with the same Cox model.

    Returns a list of :class:`StepCredit` objects, one per candidate.
    """
    fn = (closed_form_credit if method == "closed_form"
          else leave_one_step_out_credit)
    return [fn(model, X, T, E, record_ids=record_ids)
            for X in candidate_X]


__all__ = [
    "StepCredit",
    "closed_form_credit", "leave_one_step_out_credit",
    "credits_to_step_rewards", "credits_to_preference_pairs",
    "credits_from_candidates",
]
