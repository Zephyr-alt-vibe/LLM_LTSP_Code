"""Thin wrapper documenting the LT imaging preprocessing pipeline.

The CT image preprocessing for LLM-MMI-LT (Stage 1, vision side) is
implemented as a set of standalone scripts living in ``LT/``.  This
module documents the pipeline and exposes a single ``run_preprocessing``
helper that calls them in order, so that a complete end-to-end run can
be reproduced without leaving the :mod:`llm_mmi.lt` namespace.

Pipeline (mirrors ``LT/readme.md``):

    1. ``LT/preprocess.py``           — DICOM -> PNG with lung-window
                                       (level -600 HU, width 1500 HU),
                                       1 mm isotropic resampling, automated
                                       quality filtering.
    2. ``LT/build_tree.py``           — emit the per-cohort manifest
                                       ``site*.json`` linking each patient
                                       to their CT phase folders and
                                       text reports.
    3. ``LT/CAM.py``                  — encode every PNG with ResNet50 +
                                       CAM weights and dump the per-slice
                                       feature vectors; also writes
                                       ``cam_scores.json`` next to each
                                       scan folder so that
                                       :func:`llm_mmi.lt.vlm.select_keyframes_from_index`
                                       can rank slices for the VLM branch.
    4. ``LT/medical_graph_generator.py`` — optional knowledge-graph
                                       augmentation; not required for the
                                       core LLM-MMI-LT framework but
                                       useful for downstream analyses.

The output of step 2 is the file referenced by
``LT_KEYFRAME_INDEX`` in :mod:`llm_mmi.lt.config`; the CAM scores from
step 3 are picked up automatically.

The scripts retain hardcoded paths from the LT manuscript; override them
with environment variables ``LT_RAW_ROOT`` / ``LT_DST_ROOT`` before
running, or edit ``LT/preprocess.py::Config`` directly.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

LT_DIR = Path(__file__).resolve().parents[2] / "LT"


def run_preprocessing(skip_existing: bool = True) -> int:
    """Run the four preprocessing scripts sequentially.

    Returns the number of scripts that exited cleanly.  This helper is
    intentionally simple; for a production deployment, prefer driving
    each step through your scheduler and pinning the working directory.
    """
    steps = [
        ("preprocess.py",            "DICOM -> PNG"),
        ("build_tree.py",            "build manifest"),
        ("CAM.py",                   "CAM features"),
        ("medical_graph_generator.py", "knowledge graph (optional)"),
    ]
    ok = 0
    for script, label in steps:
        path = LT_DIR / script
        if not path.exists():
            print(f"[imaging] {script} not found; skipping.")
            continue
        marker = LT_DIR / f".{script}.done"
        if skip_existing and marker.exists():
            print(f"[imaging] {label}: cached ({marker.name})")
            ok += 1
            continue
        print(f"[imaging] {label}: python {path.relative_to(LT_DIR.parent)}")
        rc = subprocess.call([sys.executable, str(path)], cwd=str(LT_DIR))
        if rc == 0:
            marker.touch()
            ok += 1
        else:
            print(f"[imaging] {script} exited with status {rc}")
            break
    return ok


__all__ = ["LT_DIR", "run_preprocessing"]
