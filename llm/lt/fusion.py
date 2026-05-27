"""Multimodal feature fusion and Cox PH survival modelling for LLM-MMI-LT.

This module implements Stage 6 of the LT framework: parse the per-patient
text-branch JSON, append the VLM-derived binary CT risk, fit a Cox PH
model on the training cohort (Centre 1), and evaluate 12/18/24-month
mortality predictions in held-out validation cohorts (Centres 2 and 3).

Mirrors the multimorbidity package's Cox helper in spirit, but kept in
Python (rather than R) because LT is a Python-only pipeline (see
``LT/cox.py``).  Statistics computed:

* Harrell's C-index with asymptotic 95% CI;
* Time-dependent IPCW-AUC at 12 / 18 / 24 months;
* IPCW-Brier score with mean-Brier-based ``R^2``;
* Observed-to-expected ratio with log-Poisson 95% CI;
* Binary classification metrics at 12 months (Youden threshold).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from . import schema


# ---------------------------------------------------------------------------
# JSON parsing — recover the 15-D text vector from the teacher response
# ---------------------------------------------------------------------------
_KEY_ALIASES = {
    "behaviors/habits health grading":   "behavioral/habit health grading",  # safety
    "past medical history grading":      "medical history grading",
    "rejection risk":                    "rejection risk grading",
    "infection risk":                    "infection risk grading",
    "postoperative complication risk":   "postoperative complication risk grading",
}


def _strip_code_fence(blob: str) -> str:
    blob = blob.strip()
    if blob.startswith("```"):
        blob = re.sub(r"^```[a-zA-Z]*\n?", "", blob)
        blob = re.sub(r"```\s*$", "", blob)
    return blob.strip()


def parse_text_response(response: str) -> np.ndarray:
    """Parse a text-branch JSON response into a 15-D float vector.

    Returns NaN for missing/malformed fields; a row with any NaN is later
    filtered out in :func:`load_fusion_table`.
    """
    out = np.full(len(schema.TEXT_VARS_USE), np.nan, dtype=np.float64)
    if not isinstance(response, str) or not response.strip():
        return out
    blob = _strip_code_fence(response)
    try:
        obj = json.loads(blob)
    except json.JSONDecodeError:
        return out
    if not isinstance(obj, dict):
        return out

    obj_norm = {_KEY_ALIASES.get(k.strip(), k.strip()): v
                for k, v in obj.items()}

    for i, field in enumerate(schema.TEXT_VARS_USE):
        if field not in obj_norm:
            continue
        raw = obj_norm[field]
        if field in schema.LT_RISK_FIELDS:
            if isinstance(raw, str):
                out[i] = schema.GRADING_ENCODING.get(raw.strip().lower(), np.nan)
        else:
            try:
                out[i] = float(raw)
            except (TypeError, ValueError):
                out[i] = np.nan
    return out


# ---------------------------------------------------------------------------
# Fusion table assembly
# ---------------------------------------------------------------------------
@dataclass
class FusionTable:
    """Result of :func:`load_fusion_table`."""
    df: pd.DataFrame            # one row per patient
    X:  np.ndarray              # (n, 17) feature matrix (z-scored)
    time:  np.ndarray           # (n,) follow-up in years
    event: np.ndarray           # (n,) 0/1


def load_fusion_table(inference_path: str | Path,
                      outcome_path: str | Path,
                      drop_incomplete: bool = True,
                      ) -> FusionTable:
    """Join the inference output with the outcome table and z-score.

    Parameters
    ----------
    inference_path
        Excel/parquet emitted by :mod:`llm_mmi.lt.inference` with columns
        ``ID``, ``Response``, ``ct-derived risk``.
    outcome_path
        Excel/parquet with at minimum ``ID``, ``time`` (years to event or
        censoring), and ``event`` (1=death, 0=censored).
    drop_incomplete
        Drop rows whose 15-D text vector contains any NaN.  When False,
        the caller is responsible for downstream imputation.
    """
    inf = _read_any(inference_path)
    out = _read_any(outcome_path)

    # Two supported input layouts:
    #   (a) raw inference output emitted by :mod:`llm_mmi.lt.inference`, with
    #       columns ``ID, Response, ct-derived risk``;
    #   (b) pre-parsed Excel (e.g. ``LT/yanzheng11-*.xlsx``) where the 15
    #       text dimensions appear as named columns and the CT-derived risk
    #       lives in either ``ct-derived risk`` or the legacy ``video_risk``
    #       column.  Centre 2 encodes it as {1, 5}; Centre 3 as {0, 1}.
    if "Response" in inf.columns:
        merged = inf.merge(out, on="ID", how="inner")
        text_mat = np.stack([parse_text_response(r) for r in merged["Response"]])
    else:
        need_outcome = not all(c in inf.columns for c in ("time", "event"))
        merged = inf.merge(out, on="ID", how="inner") if need_outcome else inf.copy()
        text_mat = np.full((len(merged), len(schema.TEXT_VARS_USE)), np.nan)
        for j, field in enumerate(schema.TEXT_VARS_USE):
            if field not in merged.columns:
                continue
            raw = merged[field]
            if field in schema.LT_RISK_FIELDS:
                num = pd.to_numeric(raw, errors="coerce")
                if num.notna().any():
                    text_mat[:, j] = num.to_numpy(dtype=np.float64)
                else:
                    text_mat[:, j] = np.array([
                        schema.GRADING_ENCODING.get(
                            str(v).strip().lower(), np.nan) if pd.notna(v) else np.nan
                        for v in raw
                    ])
            else:
                text_mat[:, j] = pd.to_numeric(raw, errors="coerce").to_numpy(
                    dtype=np.float64)

    ct_col = merged.get("ct-derived risk")
    if ct_col is None:
        ct_col = merged.get("video_risk")
    if ct_col is None:
        ct_arr = np.full(len(merged), np.nan)
    else:
        ct_arr = pd.to_numeric(ct_col, errors="coerce").to_numpy()
        # Normalise to {0, 1}: Centre 2 uses {1, 5}; Centre 3 uses {0, 1}.
        if ct_arr.size and np.nanmax(ct_arr) > 1:
            ct_arr = (ct_arr >= 3).astype(np.float64)
    full = np.concatenate([text_mat, ct_arr.reshape(-1, 1)], axis=1)

    if drop_incomplete:
        mask = ~np.isnan(full).any(axis=1)
        merged = merged.loc[mask].reset_index(drop=True)
        full = full[mask]

    # Attach the parsed columns to the dataframe (handy for downstream
    # auditing and Excel-side review).
    for i, name in enumerate(schema.FUSED_VARS_USE):
        merged[name] = full[:, i]

    time = merged["time"].astype(float).to_numpy()
    event = merged["event"].astype(int).to_numpy()
    X, _, _ = _zscore(full)
    return FusionTable(df=merged, X=X, time=time, event=event)


def _read_any(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix in (".parquet", ".pq"):
        return pd.read_parquet(path)
    if path.suffix in (".csv", ".tsv"):
        sep = "\t" if path.suffix == ".tsv" else ","
        return pd.read_csv(path, sep=sep)
    return pd.read_excel(path)


def _zscore(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean_ = X.mean(axis=0)
    std_ = X.std(axis=0, ddof=0)
    std_[std_ == 0] = 1.0
    return (X - mean_) / std_, mean_, std_


# ---------------------------------------------------------------------------
# Cox PH fit (statsmodels), evaluation, and bootstrap CI
# ---------------------------------------------------------------------------
@dataclass
class CoxFit:
    params:   np.ndarray
    baseline_surv_times: np.ndarray
    baseline_surv_vals:  np.ndarray
    feature_names: list[str]


def fit_cox(ft: FusionTable) -> CoxFit:
    """Fit Cox PH on the (already z-scored) 16-D feature matrix."""
    from statsmodels.duration.hazard_regression import PHReg
    res = PHReg(endog=ft.time, exog=ft.X, status=ft.event,
                ties="breslow").fit(disp=False)
    lp = ft.X @ res.params
    bh_times, bh_surv = _baseline_survival(ft.time, ft.event, lp)
    return CoxFit(params=np.asarray(res.params),
                  baseline_surv_times=bh_times,
                  baseline_surv_vals=bh_surv,
                  feature_names=list(schema.FUSED_VARS_USE))


def _baseline_survival(time, event, lp):
    """Breslow estimator of the baseline survival curve."""
    order = np.argsort(time)
    t = time[order]
    e = event[order]
    lp = lp[order]
    risk = np.exp(lp)
    # Cumulative hazard at each unique event time.
    cum_haz, surv = 0.0, 1.0
    out_t, out_s = [], []
    rsum = risk.sum()
    for i in range(len(t)):
        ti, ei, ri = t[i], e[i], risk[i]
        if ei == 1:
            cum_haz += 1.0 / rsum if rsum > 0 else 0.0
            surv = np.exp(-cum_haz)
            out_t.append(ti); out_s.append(surv)
        rsum -= ri
    return np.asarray(out_t), np.asarray(out_s)


def predict_survival(fit: CoxFit, X: np.ndarray,
                     horizons: Iterable[float]) -> dict[float, np.ndarray]:
    """Return ``{horizon: S(horizon|x)}`` for each requested horizon."""
    lp = X @ fit.params
    risk = np.exp(lp)
    out = {}
    for h in horizons:
        idx = np.searchsorted(fit.baseline_surv_times, h, side="right") - 1
        if idx < 0:
            s0 = 1.0
        else:
            s0 = fit.baseline_surv_vals[idx]
        out[float(h)] = np.clip(s0 ** risk, 0.0, 1.0)
    return out


# ---------------------------------------------------------------------------
# Metrics: C-index, IPCW-AUC, IPCW-Brier
# ---------------------------------------------------------------------------
def harrell_c_index(time, event, risk_score) -> float:
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    risk = np.asarray(risk_score, dtype=float)
    concordant = ties = comparable = 0.0
    n = len(time)
    for i in range(n - 1):
        for j in range(i + 1, n):
            ti, tj = time[i], time[j]
            ei, ej = event[i], event[j]
            if ti == tj:
                if ei == 1 and ej == 0:
                    comparable += 1
                    concordant += (risk[i] > risk[j]) + 0.5 * (risk[i] == risk[j])
                elif ei == 0 and ej == 1:
                    comparable += 1
                    concordant += (risk[j] > risk[i]) + 0.5 * (risk[i] == risk[j])
                continue
            if ti < tj:
                if ei == 0:
                    continue
                comparable += 1
                concordant += (risk[i] > risk[j]) + 0.5 * (risk[i] == risk[j])
            else:
                if ej == 0:
                    continue
                comparable += 1
                concordant += (risk[j] > risk[i]) + 0.5 * (risk[i] == risk[j])
    return np.nan if comparable == 0 else concordant / comparable


class _KMC:
    """Kaplan–Meier estimator of the censoring distribution (for IPCW)."""

    def __init__(self, time, event):
        self.time = np.asarray(time, dtype=float)
        self.event = np.asarray(event, dtype=int)
        cens = 1 - self.event
        unique_times = np.unique(self.time)
        surv = 1.0
        vals = []
        for t in unique_times:
            at_risk = np.sum(self.time >= t)
            d = np.sum((self.time == t) & (cens == 1))
            surv *= (1.0 - d / at_risk) if at_risk > 0 else 1.0
            vals.append(surv)
        self.times = unique_times
        self.vals = np.asarray(vals)

    def G(self, t, left_limit=False):
        t = np.atleast_1d(np.asarray(t, dtype=float))
        idx = np.searchsorted(self.times, t,
                              side="left" if left_limit else "right") - 1
        out = np.ones_like(t, dtype=float)
        valid = idx >= 0
        out[valid] = self.vals[idx[valid]]
        return out


def time_dependent_auc_ipcw(time, event, risk_score, horizon, eps=1e-8) -> float:
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    risk = np.asarray(risk_score, dtype=float)
    kmc = _KMC(time, event)
    cases = (time <= horizon) & (event == 1)
    controls = time > horizon
    if cases.sum() == 0 or controls.sum() == 0:
        return np.nan
    w_case = 1.0 / np.clip(kmc.G(time[cases], left_limit=True), eps, None)
    w_ctrl = 1.0 / np.clip(kmc.G(np.repeat(horizon, controls.sum())), eps, None)
    comp = risk[cases][:, None] - risk[controls][None, :]
    gt = (comp > 0).astype(float)
    eq = (comp == 0).astype(float)
    w = w_case[:, None] * w_ctrl[None, :]
    denom = w_case.sum() * w_ctrl.sum()
    if denom <= 0:
        return np.nan
    return float(np.sum(w * (gt + 0.5 * eq)) / denom)


def brier_score_ipcw(time, event, pred_surv, horizon, eps=1e-8) -> float:
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    surv = np.asarray(pred_surv, dtype=float)
    kmc = _KMC(time, event)
    g_t = np.clip(kmc.G(np.repeat(horizon, len(time))), eps, None)
    g_i = np.clip(kmc.G(time, left_limit=True), eps, None)
    term = np.zeros(len(time))
    case_mask = (time <= horizon) & (event == 1)
    ctrl_mask = time > horizon
    term[case_mask] = (surv[case_mask]) ** 2 / g_i[case_mask]
    term[ctrl_mask] = ((1.0 - surv[ctrl_mask]) ** 2) / g_t[ctrl_mask]
    return float(np.mean(term))


# ---------------------------------------------------------------------------
# End-to-end runner: fit on train, evaluate on validation cohorts
# ---------------------------------------------------------------------------
LT_HORIZONS = (1.0, 1.5, 2.0)            # years (12 / 18 / 24 months)


def run_fusion_pipeline(train_inference: str | Path,
                        train_outcome:   str | Path,
                        validation_pairs: Iterable[tuple[str, str | Path, str | Path]] = (),
                        out_path: Optional[str | Path] = None) -> pd.DataFrame:
    """Fit Cox on the training cohort and evaluate on each validation cohort.

    Parameters
    ----------
    train_inference, train_outcome
        Inference output + outcome table for the training centre.
    validation_pairs
        Iterable of ``(centre_name, inference_path, outcome_path)``
        tuples; each is evaluated under the *fixed* training-cohort
        coefficients (no recalibration), matching LT/baseline.md §1.7.
    out_path
        Optional Excel destination for the metrics table.
    """
    train = load_fusion_table(train_inference, train_outcome)
    fit = fit_cox(train)
    rows = []
    rows.append(_evaluate("Train", train, fit))
    for name, inf_p, out_p in validation_pairs:
        valid = load_fusion_table(inf_p, out_p)
        rows.append(_evaluate(name, valid, fit))
    res = pd.DataFrame(rows)
    if out_path:
        res.to_excel(str(out_path), index=False)
    return res


def _evaluate(name: str, ft: FusionTable, fit: CoxFit) -> dict:
    surv = predict_survival(fit, ft.X, LT_HORIZONS)
    risk = {h: 1.0 - surv[h] for h in LT_HORIZONS}
    lp = ft.X @ fit.params

    row = {"cohort": name,
           "n": len(ft.time),
           "events": int(ft.event.sum()),
           "C-index": harrell_c_index(ft.time, ft.event, lp)}
    for h in LT_HORIZONS:
        row[f"AUC@{int(h*12)}m"] = time_dependent_auc_ipcw(
            ft.time, ft.event, risk[h], horizon=h)
        row[f"Brier@{int(h*12)}m"] = brier_score_ipcw(
            ft.time, ft.event, surv[h], horizon=h)
    return row


__all__ = [
    "FusionTable", "CoxFit", "LT_HORIZONS",
    "parse_text_response",
    "load_fusion_table",
    "fit_cox", "predict_survival",
    "harrell_c_index", "time_dependent_auc_ipcw", "brier_score_ipcw",
    "run_fusion_pipeline",
]
