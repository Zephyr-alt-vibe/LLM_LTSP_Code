"""Outcome regression utilities for the RFT reward.

The frozen-policy 17-d representation `x_i ∈ R^17` is mapped to a scalar
hazard through a Cox PH model

    h(t | x) = h_0(t) · exp(β · x).

This module fits ``β`` (closed form via lifelines / Breslow tie-handling),
exposes per-record reward signals usable as RFT rewards, and provides
IPCW Brier and Harrell's C-index summaries that mirror ``eval/other.R``.

Design constraints
------------------

* Pure-Python / NumPy fall-back when ``lifelines`` is unavailable; we
  ship a hand-coded Breslow Newton optimiser so the package is
  installable in container images that purposely exclude lifelines.
* Per-record contributions are computed in *closed form* from the
  partial-likelihood derivative, not by costly bootstrap or
  leave-one-out refitting.

References
----------
* Breslow, N. E.  1974.  Covariance analysis of censored survival data.
* Method.md §B.7 — aggregation operator Φ + Cox PH + Level-2 recalibration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

try:                                                # pragma: no cover
    from lifelines import CoxPHFitter
    _HAS_LIFELINES = True
except Exception:                                   # pragma: no cover
    CoxPHFitter = None                              # type: ignore[assignment]
    _HAS_LIFELINES = False


# ---------------------------------------------------------------------------
# Fitted model container
# ---------------------------------------------------------------------------
@dataclass
class CoxModel:
    """A fitted Cox PH model plus the design matrix needed to score new
    records consistently (means used for centring, std for scaling)."""

    beta: np.ndarray          # (p,)
    feature_names: Sequence[str]
    mean: np.ndarray          # (p,)
    std: np.ndarray           # (p,)
    baseline_times: np.ndarray  # (k,)
    baseline_hazard: np.ndarray  # (k,) cumulative H_0(t)

    def linear_predictor(self, X: np.ndarray) -> np.ndarray:
        """Return η_i = β·(x_i − μ) / σ (centred + scaled)."""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X[None, :]
        Xs = (X - self.mean) / np.where(self.std > 0, self.std, 1.0)
        return Xs @ self.beta

    def risk(self, X: np.ndarray) -> np.ndarray:
        return np.exp(self.linear_predictor(X))

    def survival_at(self, X: np.ndarray, t0: float) -> np.ndarray:
        """S(t0 | x) = exp(-H_0(t0) · exp(η))."""
        H0 = float(np.interp(t0, self.baseline_times, self.baseline_hazard,
                             left=0.0, right=self.baseline_hazard[-1]))
        return np.exp(-H0 * self.risk(X))


# ---------------------------------------------------------------------------
# Fitter — lifelines fast path, NumPy Breslow fall-back
# ---------------------------------------------------------------------------
def _fit_breslow_numpy(X: np.ndarray, T: np.ndarray, E: np.ndarray,
                       max_iter: int = 50, tol: float = 1e-7,
                       l2: float = 1e-4) -> np.ndarray:
    """Hand-coded Cox PH (Breslow) Newton–Raphson.

    Pure-NumPy; O(n·p²) per Newton step.  Used when ``lifelines`` is
    absent.  L2 ridge ``l2`` is added on the diagonal of the Hessian for
    numerical stability on near-collinear LLM-MMI columns.
    """
    n, p = X.shape
    beta = np.zeros(p)

    order = np.argsort(-T)                          # descending time → easy risk-set cumsum
    Xo, To, Eo = X[order], T[order], E[order]

    for _ in range(max_iter):
        eta = Xo @ beta
        eta -= eta.max()                            # numerical stability
        w = np.exp(eta)                              # (n,)
        # Cumulative sums of risk-set weights (descending time order):
        S0 = np.cumsum(w)                            # Σ_{j ∈ R(i)} w_j
        S1 = np.cumsum(w[:, None] * Xo, axis=0)      # (n,p)
        # Mean covariate within risk set:
        mean_x = S1 / S0[:, None]
        # Score (gradient): Σ_{i: E=1} (x_i − x̄(R_i))
        score = ((Xo - mean_x) * Eo[:, None]).sum(axis=0)
        # Hessian: Σ_{i: E=1} (S2_i/S0_i − x̄ x̄ᵀ)
        # Compute S2 efficiently as the cumulative outer product of w·x.
        S2 = np.cumsum(
            (w[:, None, None] * Xo[:, :, None] * Xo[:, None, :]),
            axis=0,
        )                                            # (n,p,p)
        hess = np.zeros((p, p))
        events = np.where(Eo == 1)[0]
        for i in events:
            cov_i = S2[i] / S0[i] - np.outer(mean_x[i], mean_x[i])
            hess += cov_i
        hess += l2 * np.eye(p)
        try:
            step = np.linalg.solve(hess, score)
        except np.linalg.LinAlgError:               # pragma: no cover
            break
        beta += step
        if np.max(np.abs(step)) < tol:
            break
    return beta


def _breslow_baseline(X: np.ndarray, T: np.ndarray, E: np.ndarray,
                      beta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return Breslow cumulative baseline hazard at the unique event times."""
    eta = X @ beta
    w = np.exp(eta - eta.max())
    order = np.argsort(T)
    To, Eo, wo = T[order], E[order], w[order]
    # For each event time, Σ_{j: T_j ≥ t} w_j; sweep from the tail.
    rev_cumw = np.cumsum(wo[::-1])[::-1]
    unique_t, idx = np.unique(To[Eo == 1], return_inverse=False), None
    haz = []
    for t in unique_t:
        risk_set = rev_cumw[np.searchsorted(To, t, side="left")]
        d_t = np.sum((To == t) & (Eo == 1))
        haz.append(d_t / max(risk_set, 1e-12))
    return unique_t, np.cumsum(haz)


def fit_cox(X: np.ndarray, T: np.ndarray, E: np.ndarray,
            feature_names: Sequence[str],
            l2: float = 1e-4) -> CoxModel:
    """Fit a Cox PH model.  Standardises X column-wise before fitting."""
    X = np.asarray(X, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    E = np.asarray(E, dtype=np.int64)

    mask = (~np.isnan(X).any(axis=1)) & (~np.isnan(T)) & (~np.isnan(E))
    X, T, E = X[mask], T[mask], E[mask]
    if X.shape[0] < 50:
        raise ValueError(f"too few complete cases to fit Cox: n={X.shape[0]}")

    mean = X.mean(axis=0)
    std  = X.std(axis=0)
    std_safe = np.where(std > 0, std, 1.0)
    Xs = (X - mean) / std_safe

    if _HAS_LIFELINES:                              # pragma: no cover
        import pandas as pd
        df = pd.DataFrame(Xs, columns=list(feature_names))
        df["time"], df["event"] = T, E
        cph = CoxPHFitter(penalizer=l2)
        cph.fit(df, duration_col="time", event_col="event",
                show_progress=False)
        beta = cph.params_.to_numpy()
    else:
        beta = _fit_breslow_numpy(Xs, T, E, l2=l2)

    bh_times, bh_cum = _breslow_baseline(Xs, T, E, beta)
    return CoxModel(beta=beta, feature_names=list(feature_names),
                    mean=mean, std=std_safe,
                    baseline_times=bh_times, baseline_hazard=bh_cum)


# ---------------------------------------------------------------------------
# Per-record reward signals (the RFT reward)
# ---------------------------------------------------------------------------
def per_record_partial_loglik(model: CoxModel,
                              X: np.ndarray,
                              T: np.ndarray,
                              E: np.ndarray) -> np.ndarray:
    """Return the per-record contribution to the Cox partial log-likelihood.

    For event records, the contribution is

        ℓ_i = η_i − log Σ_{j ∈ R(i)} exp(η_j),

    for censored records ℓ_i = 0 (they enter only via the risk set).
    A *higher* ℓ_i means the model gives this participant a higher
    relative hazard amongst still-at-risk individuals — i.e. the
    representation better explains the observed event.  This is the
    natural RFT reward signal: maximise the partial likelihood of the
    *observed* trajectory under the *student-produced* features.
    """
    eta = model.linear_predictor(X)
    n = len(eta)
    out = np.zeros(n)
    order = np.argsort(-T)
    rev = np.argsort(order)
    eta_o, T_o, E_o = eta[order], T[order], E[order]
    # log-sum-exp cumulative in descending time:
    log_cum = np.zeros(n)
    m = -np.inf
    s = 0.0
    for i in range(n):
        new_m = max(m, eta_o[i])
        s = np.exp(m - new_m) * s + np.exp(eta_o[i] - new_m)
        m = new_m
        log_cum[i] = m + np.log(s)
    out_o = np.where(E_o == 1, eta_o - log_cum, 0.0)
    return out_o[rev]


def harrell_c_index(model: CoxModel,
                    X: np.ndarray, T: np.ndarray, E: np.ndarray) -> float:
    """Harrell's concordance.  O(n²); use a subsample for n > 50k."""
    eta = model.linear_predictor(X)
    n = len(eta)
    num = den = 0.0
    for i in range(n):
        if E[i] != 1:
            continue
        for j in range(n):
            if T[j] <= T[i] or i == j:
                continue
            den += 1
            if eta[i] > eta[j]:
                num += 1
            elif eta[i] == eta[j]:
                num += 0.5
    return num / den if den > 0 else float("nan")


def ipcw_brier(model: CoxModel,
               X: np.ndarray, T: np.ndarray, E: np.ndarray,
               horizon: float) -> float:
    """IPCW Brier score at the given horizon (matches eval/other.R)."""
    # Kaplan–Meier estimate of the censoring distribution G(t).
    order = np.argsort(T)
    To, Eo = T[order], E[order]
    at_risk = len(To)
    G = 1.0
    G_times = [0.0]; G_vals = [1.0]
    for i, t in enumerate(To):
        if Eo[i] == 0:                              # censoring event
            G *= 1 - 1 / max(at_risk, 1)
        at_risk -= 1
        G_times.append(float(t)); G_vals.append(G)
    G_times = np.array(G_times); G_vals = np.array(G_vals)

    def G_at(t):
        return float(np.interp(t, G_times, G_vals, right=G_vals[-1]))

    pred = 1 - model.survival_at(X, horizon)
    bs = 0.0; n = 0
    for i in range(len(T)):
        ti, ei = T[i], E[i]
        if ti <= horizon and ei == 1:
            w = 1.0 / max(G_at(ti), 1e-6)
            bs += w * (1 - pred[i]) ** 2
            n += 1
        elif ti > horizon:
            w = 1.0 / max(G_at(horizon), 1e-6)
            bs += w * (0 - pred[i]) ** 2
            n += 1
    return bs / max(n, 1)


# ---------------------------------------------------------------------------
# Reward shaping
# ---------------------------------------------------------------------------
def record_reward(model: CoxModel,
                  X: np.ndarray, T: np.ndarray, E: np.ndarray,
                  baseline_loglik: Optional[np.ndarray] = None) -> np.ndarray:
    """Reward used by the RFT loop.

    By default the reward is the per-record partial log-likelihood
    (centred at zero).  If a ``baseline_loglik`` (computed on the
    teacher's outputs) is supplied, the reward is the *advantage*

        r_i = ℓ_i(student) − ℓ_i(teacher)

    which is the natural target for DPO-style preference pairs:
    student's sample is preferred over teacher's when r_i > 0.
    """
    ll = per_record_partial_loglik(model, X, T, E)
    if baseline_loglik is not None:
        return ll - baseline_loglik
    return ll - ll.mean()


__all__ = [
    "CoxModel", "fit_cox",
    "per_record_partial_loglik", "harrell_c_index", "ipcw_brier",
    "record_reward",
]
