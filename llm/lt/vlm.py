"""VLM zero-shot CT keyframe branch for LLM-MMI-LT (Stage 5b).

The VLM branch is intentionally lightweight: a single frozen pass of the
multimodal policy over the CAM-selected keyframes (output of
``LT/CAM.py`` / ``LT/build_tree.py``) yielding a four-step CoT JSON whose
final field is a binary CT-derived risk classification.  This binary
feature is concatenated with the 15-D text branch in
:mod:`llm_mmi.lt.fusion`.

The prompt template mirrors ``LT/baseline.md`` §1.6 (eMethods 2).
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from . import config


# ---------------------------------------------------------------------------
# Prompt template (the same across all preset configurations because the
# VLM branch is a single self-contained zero-shot CoT call).
# ---------------------------------------------------------------------------
VLM_PROMPT = (
    "You are an advanced medical imaging AI assistant, serving as a "
    "process supervision model for Multiple Instance Learning (MIL) "
    "diagnostic tasks. You possess a profound understanding of "
    "long-term survival rates following lung transplantation, post-"
    "operative risk stratification, biomarkers, and clinical "
    "manifestations during the recovery phase. You are particularly "
    "adept at providing personalized prognostic predictions for "
    "physicians by analyzing longitudinal imaging transitions during "
    "hospitalization.\n\n"
    "In this task, you will analyze a patient case presented as an "
    "instance \"bag\" containing multiple medical images. Your "
    "objective is to determine whether the patient's prognostic risk "
    "is high or low and to evaluate the diagnostic reasoning process "
    "step-by-step:\n"
    "1. Global Screening: Initial overview of all images in the "
    "patient bag.\n"
    "2. Instance Selection: Identify which images show potential "
    "abnormalities (ADA-selected keyframes based on clinical text "
    "relevance).\n"
    "3. Detailed Analysis: Examine selected suspicious instances for "
    "specific findings (consolidation patterns, ground-glass "
    "opacities, pleural effusion, airway changes, signs of chronic "
    "lung allograft dysfunction, and interval changes between time "
    "points).\n"
    "4. Diagnostic Synthesis: Integrate findings across instances and "
    "time points to form the final prognostic risk assessment.\n\n"
    "The format of your answer is JSON, please do not give any "
    "additional output, please refer to the following format:\n"
    "```json\n"
    "{\n"
    '  "Step1": string,\n'
    '  "Global Screening": int,\n'
    '  "Step2": string,\n'
    '  "Instance Selection": int,\n'
    '  "Step3": string,\n'
    '  "Detailed Analysis": int,\n'
    '  "Step4": string,\n'
    '  "Diagnostic Synthesis": int,\n'
    '  "CT-derived risk": string  // \"high-risk\" or \"low-risk\"\n'
    "}\n"
    "```\n"
    "The keyframes are presented in temporal order (preoperative -> "
    "postoperative -> post-discharge); each carries a metadata tag "
    "indicating its acquisition time point."
)


# ---------------------------------------------------------------------------
# Keyframe selection — picks the top-K CAM keyframes per phase
# ---------------------------------------------------------------------------
# Phase ordering follows the LT preprocess pipeline (``LT/preprocess.py``).
_PHASE_ORDER = ("术前", "出院前最新", "出院后第一次")
_PHASE_LABEL = {
    "术前":            "Pre-op",
    "出院前最新":      "Pre-discharge",
    "出院后第一次":    "Post-discharge",
}


def select_keyframes_from_index(patient_record: dict,
                                topk: int = config.VLM_TOPK_KEYFRAMES,
                                ) -> List[dict]:
    """Pick the top-``K`` CAM-ranked keyframes per phase for one patient.

    Parameters
    ----------
    patient_record
        One entry of the ``site*.json`` manifest emitted by
        ``LT/build_tree.py`` + ``LT/CAM.py``.  Expected shape::

            {
              "CT": {
                "术前":            {<sub>: {"file_path": ..., "vector_path": ..., "cam_score": ...}},
                "出院前最新":      {...},
                "出院后第一次":    {...}
              },
              "Report": {...}
            }

        Each scan sub-folder must contain ``.png`` files; CAM scores per
        slice are read from a sibling ``cam_scores.json`` when present,
        otherwise slices are kept in directory order.
    topk
        Number of keyframes to retain per phase.

    Returns
    -------
    keyframes : list of dict
        Each entry has ``path`` (absolute PNG path) and ``tag``
        (e.g. ``"Pre-op Slice 45"``) in temporal order.
    """

    ct = patient_record.get("CT", {}) if isinstance(patient_record, dict) else {}
    out: List[dict] = []
    for phase in _PHASE_ORDER:
        scans = ct.get(phase, {})
        if not scans:
            continue
        # Collect every PNG below this phase.
        slices: List[tuple[float, str]] = []
        for sub_name, scan_info in scans.items():
            scan_dir = scan_info.get("file_path") if isinstance(scan_info, dict) else None
            if not scan_dir or not os.path.isdir(scan_dir):
                continue
            cam_path = os.path.join(scan_dir, "cam_scores.json")
            cam_lookup: dict = {}
            if os.path.exists(cam_path):
                try:
                    with open(cam_path, "r", encoding="utf-8") as f:
                        cam_lookup = json.load(f)
                except (OSError, json.JSONDecodeError):
                    cam_lookup = {}
            for fname in sorted(os.listdir(scan_dir)):
                if not fname.endswith(".png"):
                    continue
                if fname.endswith(("_heatmap.png", "_overlay.png")):
                    continue
                score = cam_lookup.get(fname)
                if score is None:
                    # No score -> deterministic fallback by filename order.
                    score = -float(len(slices))
                slices.append((float(score), os.path.join(scan_dir, fname)))
        if not slices:
            continue
        slices.sort(key=lambda t: -t[0])
        for rank, (_, path) in enumerate(slices[:topk]):
            slice_no_match = re.search(r"(\d+)", os.path.basename(path))
            slice_no = slice_no_match.group(1) if slice_no_match else str(rank + 1)
            out.append({
                "phase": phase,
                "path":  path,
                "tag":   f"{_PHASE_LABEL[phase]} Slice {slice_no}",
            })
    return out


# ---------------------------------------------------------------------------
# OpenAI-compatible multimodal message builder
# ---------------------------------------------------------------------------
def _encode_image_to_data_url(path: str) -> str:
    """Encode a PNG file as a base64 ``data:`` URI for OpenAI-style payloads."""
    mime, _ = mimetypes.guess_type(path)
    if not mime:
        mime = "image/png"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def build_vlm_messages(keyframes: Sequence[dict],
                       prompt_text: Optional[str] = None,
                       narrative_excerpt: Optional[str] = None,
                       ) -> list[dict]:
    """Build the chat-completions message list for the VLM branch.

    Parameters
    ----------
    keyframes
        Output of :func:`select_keyframes_from_index`; ordered temporally.
    prompt_text
        Override the default :data:`VLM_PROMPT`.
    narrative_excerpt
        Optional short clinical context (e.g. the discharge summary
        header) appended to the user message; useful when the VLM
        backbone supports cross-modal grounding but the text branch is
        running on a separate cheaper LLM.
    """

    text = prompt_text or VLM_PROMPT
    user_content: list[dict] = [{"type": "text", "text": text}]

    for kf in keyframes:
        user_content.append({"type": "text",
                             "text": f"[{kf['tag']}]"})
        try:
            data_url = _encode_image_to_data_url(kf["path"])
        except OSError:
            continue
        user_content.append({
            "type": "image_url",
            "image_url": {"url": data_url, "detail": "low"},
        })

    if narrative_excerpt:
        user_content.append({"type": "text",
                             "text": "Brief clinical context: "
                                     + narrative_excerpt[:1000]})

    return [{"role": "user", "content": user_content}]


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------
_BINARY_MAP = {
    "high-risk": 1, "high risk": 1, "high": 1,
    "low-risk": 0,  "low risk": 0,  "low": 0,
}


def parse_vlm_response(response: str) -> dict:
    """Extract the four-step CoT and binary CT-derived risk.

    Returns ``{"cot": dict-or-None, "ct_risk_label": str-or-None,
    "ct_risk_binary": 0/1 or None}``.
    """
    if not isinstance(response, str) or not response.strip():
        return {"cot": None, "ct_risk_label": None, "ct_risk_binary": None}

    blob = response.strip()
    # Strip code fence
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", blob, re.DOTALL)
    if m:
        blob = m.group(1)
    try:
        obj = json.loads(blob)
    except json.JSONDecodeError:
        return {"cot": None, "ct_risk_label": None, "ct_risk_binary": None}

    if not isinstance(obj, dict):
        return {"cot": None, "ct_risk_label": None, "ct_risk_binary": None}

    label = obj.get("CT-derived risk")
    if isinstance(label, str):
        normed = label.strip().lower()
        binary = _BINARY_MAP.get(normed)
    else:
        normed = None
        binary = None
    return {"cot": obj, "ct_risk_label": normed, "ct_risk_binary": binary}


# ---------------------------------------------------------------------------
# Backbone dispatch helper
# ---------------------------------------------------------------------------
def get_vlm_backbone() -> dict:
    """Return the backbone descriptor used by the VLM branch.

    Raises ``RuntimeError`` if the selected backbone does not advertise
    ``supports_vision``.  Falls back to ``ACTIVE_BACKBONE`` when
    ``LT_VLM_BACKBONE`` is unset.
    """
    name = config.VLM_BACKBONE
    if name not in config.POLICY_BACKBONES:
        raise RuntimeError(f"unknown VLM backbone: {name}")
    desc = config.POLICY_BACKBONES[name]
    if not desc.get("supports_vision"):
        raise RuntimeError(
            f"backbone {name!r} does not advertise supports_vision=True; "
            "set LT_VLM_BACKBONE to a multimodal backbone "
            "(e.g. internvl3-14b, gpt-5-nano, qwen2.5-vl-72b)."
        )
    return desc


__all__ = [
    "VLM_PROMPT",
    "select_keyframes_from_index",
    "build_vlm_messages",
    "parse_vlm_response",
    "get_vlm_backbone",
]
