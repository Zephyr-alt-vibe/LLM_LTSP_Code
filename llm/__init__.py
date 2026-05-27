"""LLM-MMI: Large-Language-Model-derived Multimorbidity Index.

A six-stage frozen-policy reasoning pipeline that converts harmonised
participant-level health records into a 17-dimensional multimorbidity
representation suitable for Cox proportional-hazards mortality prediction.

See ``Method.md`` (section "B. Methods") for the methodology and
``README.md`` for the methodology-to-code mapping.

Sub-modules:

* :mod:`llm_mmi.config`    — configuration knobs, ablation presets, and
  backbone catalogue (Stages 2, 5);
* :mod:`llm_mmi.schema`    — 18-node DAG output schemas (Stage 3);
* :mod:`llm_mmi.prompt`    — four-layer prompt assembly ``P = R || K ||
  S_cot || S_out || n`` (Stage 4);
* :mod:`llm_mmi.inference` — multi-threaded population-scale runner
  (Stages 2 + 5).
"""

from . import config, prompt, schema  # noqa: F401

__all__ = ["config", "prompt", "schema", "lt"]
__version__ = "0.1.0"

# The LT (lung-transplant) instantiation of the framework lives in
# :mod:`llm_mmi.lt`; see that subpackage for the LT-specific config,
# schema, prompt, and fusion modules.
