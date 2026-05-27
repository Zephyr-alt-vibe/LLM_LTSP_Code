"""LLM-MMI-LT: LLM-MMI framework instantiated for lung transplantation.

This subpackage applies the six-stage frozen-policy LLM-MMI pipeline
(:mod:`llm_mmi`) to the *Lung Transplant Survival Prediction* (LTSP) task.
The same scaffolding is reused — harmoniser ``g``, frozen policy ``pi_theta``,
DAG decomposition, four-layer prompt ``P``, deterministic single-pass
sampling, and a linear Cox head — but the per-stage content is adapted to
the LT setting:

* Stage 1 ``g`` — concatenate longitudinal CT reports + biochemistry +
  discharge summaries with section/time delimiters (see ``LT/preprocess.py``,
  ``LT/build_tree.py``);
* Stage 2 ``pi_theta`` — frozen InternVL3-14B (and other text/vision LLMs);
* Stage 3 — 16-node DAG with one anchor, ten organ-age sub-tasks, four
  LT-specific postoperative risk sub-tasks, and one integrative node;
* Stage 4 — four-layer prompt re-instantiated with the lung-transplant
  expert persona and graft–host coupling knowledge;
* Stage 5 — greedy (temperature ``T=0``) sampling, plus a parallel VLM
  branch that runs zero-shot CoT over CAM-selected keyframes from
  ``LT/CAM.py`` to emit a binary CT-derived risk;
* Stage 6 — concatenate the 15 text-derived features with the 1 CT-derived
  binary into a 16-D vector and fit a Cox PH model for 12/18/24-month
  mortality (see :mod:`llm_mmi.lt.fusion`).

Sub-modules:

* :mod:`llm_mmi.lt.config`    — backbone catalogue + ablation presets;
* :mod:`llm_mmi.lt.schema`    — 16-node DAG output schema;
* :mod:`llm_mmi.lt.prompt`    — four-layer prompt for the text branch;
* :mod:`llm_mmi.lt.vlm`       — VLM zero-shot CT keyframe prompt;
* :mod:`llm_mmi.lt.inference` — population-scale runner driving both
  branches and merging their outputs;
* :mod:`llm_mmi.lt.fusion`    — 16-D fusion and Cox PH survival modelling
  with 12/18/24-month evaluation;
* :mod:`llm_mmi.lt.imaging`   — thin wrapper documenting the LT/ imaging
  preprocessing pipeline (DICOM → PNG → CAM keyframes).
"""

from . import config, prompt, schema, vlm  # noqa: F401

__all__ = ["config", "prompt", "schema", "vlm"]
__version__ = "0.1.0"
