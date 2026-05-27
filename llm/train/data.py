"""Data interfaces for the outcome-grounded RFT pipeline.

The pipeline operates on records of the form

    Record = (id, narrative, teacher_response, x_vector, time, event)

where

* ``narrative``        — the harmonised English narrative ``n_i``
                         (Stage 1 of Method.md);
* ``teacher_response`` — raw JSON+CoT string produced by the frozen
                         policy ``π_θ`` (the ``Response`` column of
                         ``output.xlsx``, see README §9);
* ``x_vector``         — 17-d numeric representation parsed out of
                         ``teacher_response`` (matches the column order
                         of ``eval/other.R::vars_use``);
* ``time``, ``event``  — right-censored survival outcome.

Two adapters are provided:

* :class:`LLMMMIExcelAdapter` — reads the Excel artefacts produced by
  ``llm_mmi.inference`` and the canonical UKB / NHANES outcome tables;
* :class:`GenericTabularAdapter` — parquet / CSV based generic loader
  for users whose narrative table and outcome table live elsewhere.

Both yield :class:`Record` objects through an iterable / streaming
interface so that 1M–10M-row corpora can be processed without holding
the whole dataset in memory.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 17-field schema (must stay in sync with eval/other.R::vars_use)
# ---------------------------------------------------------------------------
ORGAN_AGE_FIELDS: List[str] = [
    "cardiovascular system age",
    "digestive system age",
    "respiratory system age",
    "endocrine/metabolic system age",
    "nervous system age",
    "hematologic system age",
    "musculoskeletal/motor system age",
    "urinary system age",
    "immune system age",
    "frailty age",
]

GRADING_FIELDS: List[str] = [
    "psychological health grading",
    "dietary health grading",
    "behavioral/habit health grading",
    "income grading",
    "family history grading",
    "medical history",
]

INTEGRATIVE_FIELD = "overall biological age"

VARS_USE: List[str] = ORGAN_AGE_FIELDS + GRADING_FIELDS + [INTEGRATIVE_FIELD]

# Higher value ⇒ higher hazard, so "high-risk" gets the largest code.
GRADING_ENCODING = {
    "low-risk":         1,
    "medium-low-risk":  2,
    "medium-risk":      3,
    "medium-high-risk": 4,
    "high-risk":        5,
}

# The per-task CoT keys emitted by ``schema.render_json_schema``.
_CoT_KEY_RE = re.compile(r"^inference process(?:\s+\d+|\s+disease)?$", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Core record dataclass
# ---------------------------------------------------------------------------
@dataclass
class Record:
    """One participant + their teacher response + outcome."""

    id: str
    narrative: str
    teacher_response: Optional[str] = None
    x_vector: Optional[np.ndarray] = None
    cot_steps: List[dict] = field(default_factory=list)
    time: Optional[float] = None
    event: Optional[int] = None

    def has_x(self) -> bool:
        return self.x_vector is not None and not np.any(np.isnan(self.x_vector))

    def has_outcome(self) -> bool:
        return self.time is not None and self.event is not None


# ---------------------------------------------------------------------------
# JSON / CoT parsing
# ---------------------------------------------------------------------------
def _strip_code_fence(blob: str) -> str:
    """Strip the ```json ... ``` fence the prompt asks the LLM to emit."""
    blob = blob.strip()
    if blob.startswith("```"):
        blob = re.sub(r"^```[a-zA-Z]*\n?", "", blob)
        blob = re.sub(r"```\s*$", "", blob)
    return blob.strip()


def parse_teacher_response(response: str) -> tuple[np.ndarray, List[dict]]:
    """Parse a teacher response string into (x_vector, cot_steps).

    Returns ``(x_vector, cot_steps)`` where

    * ``x_vector``  — float64 array of length ``len(VARS_USE)``; missing
                      values are NaN;
    * ``cot_steps`` — list of ``{"key": …, "field": …, "rationale": …}``
                      one per ``"inference process N"`` key recovered.
    """
    if not isinstance(response, str) or not response.strip():
        return np.full(len(VARS_USE), np.nan), []

    blob = _strip_code_fence(response)
    try:
        obj = json.loads(blob)
    except json.JSONDecodeError:
        return np.full(len(VARS_USE), np.nan), []

    if not isinstance(obj, dict):
        return np.full(len(VARS_USE), np.nan), []

    # Normalise key names: the policy occasionally emits trailing spaces
    # or the historical "behaviors/habits …" vs "behavioral/habit …" pair.
    key_aliases = {
        "behaviors/habits health grading": "behavioral/habit health grading",
        "past medical history grading":    "medical history",
    }
    obj_norm = {key_aliases.get(k.strip(), k.strip()): v for k, v in obj.items()}

    # 1) Extract the 17-d value vector.
    x = np.full(len(VARS_USE), np.nan, dtype=np.float64)
    for i, field_name in enumerate(VARS_USE):
        if field_name not in obj_norm:
            continue
        raw = obj_norm[field_name]
        if field_name in GRADING_FIELDS:
            if isinstance(raw, str):
                x[i] = GRADING_ENCODING.get(raw.strip().lower(), np.nan)
        else:
            try:
                x[i] = float(raw)
            except (TypeError, ValueError):
                x[i] = np.nan

    # 2) Extract per-step CoT.  Each "inference process N" rationale is
    #    attributed to the *next* value field in the schema, matching
    #    the layout produced by ``schema.render_json_schema``.
    keys_ordered = list(obj_norm.keys())
    value_fields_iter = iter([f for f in VARS_USE if f in obj_norm])
    cot_steps: List[dict] = []
    for k in keys_ordered:
        if not isinstance(k, str) or not _CoT_KEY_RE.match(k):
            continue
        rationale = obj_norm[k]
        if not isinstance(rationale, str):
            continue
        if k.strip().lower() == "inference process disease":
            cot_steps.append({"key": k, "field": "_anchor", "rationale": rationale})
        else:
            try:
                target_field = next(value_fields_iter)
            except StopIteration:
                target_field = None
            cot_steps.append({"key": k, "field": target_field, "rationale": rationale})

    return x, cot_steps


# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------
class _BaseAdapter(Iterable[Record]):
    """Iterable of :class:`Record`.  Subclasses must implement ``__iter__``."""

    def __iter__(self) -> Iterator[Record]:  # pragma: no cover - interface
        raise NotImplementedError

    def to_frame(self, max_rows: Optional[int] = None) -> pd.DataFrame:
        """Materialise the iterable into a pandas DataFrame.

        Use only for ≲1M rows; for 10M-row corpora iterate directly.
        """
        rows: List[dict] = []
        for n, rec in enumerate(self):
            if max_rows is not None and n >= max_rows:
                break
            row = {"id": rec.id, "narrative": rec.narrative,
                   "time": rec.time, "event": rec.event}
            if rec.x_vector is not None:
                for i, name in enumerate(VARS_USE):
                    row[name] = rec.x_vector[i]
            rows.append(row)
        return pd.DataFrame(rows)


class LLMMMIExcelAdapter(_BaseAdapter):
    """Read the artefacts produced by :mod:`llm_mmi.inference`.

    Parameters
    ----------
    narrative_path : path-like
        Excel/parquet with at least ``ID``, ``input`` columns
        (the ``LLM_MMI_DATA`` artefact).
    response_path : path-like, optional
        Excel/parquet with at least ``ID``, ``Response`` columns
        (the ``LLM_MMI_OUTPUT`` artefact); if ``None``, records will
        have ``teacher_response = None``.
    outcome_path : path-like, optional
        Excel/parquet with ``ID``, ``time``, ``event`` columns.  If
        ``None``, records have ``time = event = None``.
    """

    def __init__(self,
                 narrative_path: str | Path,
                 response_path: Optional[str | Path] = None,
                 outcome_path: Optional[str | Path] = None,
                 chunksize: int = 50_000):
        self.narrative_path = Path(narrative_path)
        self.response_path  = Path(response_path) if response_path else None
        self.outcome_path   = Path(outcome_path) if outcome_path else None
        self.chunksize      = chunksize

    @staticmethod
    def _read_any(path: Path) -> pd.DataFrame:
        if path.suffix in (".parquet", ".pq"):
            return pd.read_parquet(path)
        if path.suffix in (".csv", ".tsv"):
            return pd.read_csv(path, sep="\t" if path.suffix == ".tsv" else ",")
        return pd.read_excel(path)

    def __iter__(self) -> Iterator[Record]:
        narr = self._read_any(self.narrative_path).set_index("ID")
        if self.response_path is not None:
            resp = self._read_any(self.response_path).set_index("ID")
            resp = resp[~resp.index.duplicated(keep="last")]
        else:
            resp = None
        if self.outcome_path is not None:
            out = self._read_any(self.outcome_path).set_index("ID")
        else:
            out = None

        for pid, row in narr.iterrows():
            narrative = str(row.get("input", ""))
            teacher_response = None
            x_vec, cot = np.full(len(VARS_USE), np.nan), []
            if resp is not None and pid in resp.index:
                teacher_response = str(resp.loc[pid, "Response"])
                x_vec, cot = parse_teacher_response(teacher_response)
            time = event = None
            if out is not None and pid in out.index:
                time = float(out.loc[pid, "time"])
                event = int(out.loc[pid, "event"])
            yield Record(id=str(pid),
                         narrative=narrative,
                         teacher_response=teacher_response,
                         x_vector=x_vec,
                         cot_steps=cot,
                         time=time, event=event)


class GenericTabularAdapter(_BaseAdapter):
    """Adapter for arbitrary tabular sources, expecting one joined frame.

    The frame must expose at minimum columns
    ``id_col``, ``narrative_col``, ``time_col``, ``event_col``.
    ``response_col`` is optional.
    """

    def __init__(self, frame: pd.DataFrame,
                 id_col: str = "id",
                 narrative_col: str = "narrative",
                 response_col: Optional[str] = "response",
                 time_col: str = "time",
                 event_col: str = "event"):
        self.frame = frame
        self.id_col = id_col
        self.narrative_col = narrative_col
        self.response_col = response_col
        self.time_col = time_col
        self.event_col = event_col

    def __iter__(self) -> Iterator[Record]:
        for _, row in self.frame.iterrows():
            resp = (str(row[self.response_col])
                    if self.response_col and self.response_col in self.frame.columns
                    else None)
            if resp is not None:
                x, cot = parse_teacher_response(resp)
            else:
                x, cot = np.full(len(VARS_USE), np.nan), []
            yield Record(
                id=str(row[self.id_col]),
                narrative=str(row[self.narrative_col]),
                teacher_response=resp,
                x_vector=x,
                cot_steps=cot,
                time=float(row[self.time_col]) if self.time_col in self.frame.columns else None,
                event=int(row[self.event_col]) if self.event_col in self.frame.columns else None,
            )


# ---------------------------------------------------------------------------
# Tokeniser-side adapter for HF training
# ---------------------------------------------------------------------------
def to_chat_example(rec: Record,
                    system_prompt: Optional[str] = None) -> dict:
    """Return a single training example in HF chat-template format.

    The user turn is the narrative; the assistant turn is the teacher
    response.  Records without ``teacher_response`` are skipped by the
    trainer (filtered upstream).
    """
    messages: List[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": rec.narrative})
    if rec.teacher_response:
        messages.append({"role": "assistant", "content": rec.teacher_response})
    return {"id": rec.id, "messages": messages}


def stream_chat_examples(adapter: _BaseAdapter,
                         system_prompt: Optional[str] = None,
                         require_teacher: bool = True) -> Iterator[dict]:
    """Yield HF-ready chat examples from any adapter."""
    for rec in adapter:
        if require_teacher and not rec.teacher_response:
            continue
        yield to_chat_example(rec, system_prompt=system_prompt)


__all__ = [
    "VARS_USE", "ORGAN_AGE_FIELDS", "GRADING_FIELDS", "INTEGRATIVE_FIELD",
    "GRADING_ENCODING",
    "Record", "parse_teacher_response",
    "LLMMMIExcelAdapter", "GenericTabularAdapter",
    "to_chat_example", "stream_chat_examples",
]
