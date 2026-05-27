"""LLM-MMI-LT inference configuration.

Stage-wise mapping (see ``method.tex``, §The LLM-MMI-LT framework):

    Stage 2 — frozen policy LLM/VLM      -> POLICY_BACKBONES, ACTIVE_BACKBONE
    Stage 3 — DAG decomposition          -> ABLATION["decomposition_k"]
    Stage 4 — four-layer prompt schema   -> ABLATION["use_*_layer"]
    Stage 5 — deterministic sampling     -> TEMPERATURE, ABLATION["n_samples"]
    Stage 5b — VLM zero-shot CT branch   -> VLM_BACKBONE, VLM_TOPK_KEYFRAMES
    Stage 6 — fusion + Cox PH            -> handled in :mod:`llm_mmi.lt.fusion`

Ablation presets follow the same convention as the multimorbidity package
(``LLM_MMI_PRESET``) but the per-preset content is adapted to the LT
sub-task list.  All knobs are environment-variable overridable.
"""

import os


# ---------------------------------------------------------------------------
# Stage 2: frozen policy backbones (text + multimodal)
# ---------------------------------------------------------------------------
# All backbones are accessed through an OpenAI-compatible interface.  The
# primary backbone for both text and vision branches is InternVL3-14B,
# deployed locally on 8x A100 (80 GB) with INT8 weight-only quantisation
# (see ``method.tex``, §Stage 2).  Cross-backbone transportability
# experiments swap only the ``model`` field while keeping the same prompt
# template and Cox head.
POLICY_BACKBONES = {
    "internvl3-14b": {
        "model":    "InternVL3-14B",
        "base_url": os.environ.get(
            "LT_INTERNVL_BASE_URL",
            "http://localhost:8000/v1",     # local vLLM/lmdeploy server
        ),
        "supports_vision": True,
    },
    "deepseek-v3": {
        "model":    "deepseek-chat",
        "base_url": "https://api.deepseek.com/v1",
        "supports_vision": False,
    },
    "gpt-5-nano": {
        "model":    "gpt-5-nano",
        "base_url": "https://api.openai.com/v1",
        "supports_vision": True,
    },
    "qwen2.5-vl-72b": {
        "model":    "qwen2.5-vl-72b-instruct",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "supports_vision": True,
    },
    "gemini-2.5-flash": {
        "model":    "gemini-2.5-flash-preview-05-20",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "supports_vision": True,
    },
}

ACTIVE_BACKBONE = os.environ.get("LT_BACKBONE", "internvl3-14b")

# A separate variable is exposed for the VLM branch so the text branch can
# run on a cheap text-only model (e.g. DeepSeek-V3) while the CT branch
# still uses InternVL3-14B locally.  When unset, both branches share
# ``ACTIVE_BACKBONE`` and a runtime check enforces ``supports_vision``.
VLM_BACKBONE = os.environ.get("LT_VLM_BACKBONE", ACTIVE_BACKBONE)


# ---------------------------------------------------------------------------
# Stage 5: deterministic single-pass sampling
# ---------------------------------------------------------------------------
TEMPERATURE     = 0.0
TIMEOUT_SECONDS = 600


# ---------------------------------------------------------------------------
# Stage 5b: VLM keyframe budget
# ---------------------------------------------------------------------------
# Attention-Driven Aggregation keeps the top-K keyframes per acquisition
# time point (preoperative / postoperative / post-discharge).  The default
# of K=8 fits comfortably inside the InternVL3-14B 128K context window
# alongside the clinical narrative (3,000–15,000 tokens).
VLM_TOPK_KEYFRAMES = int(os.environ.get("LT_VLM_TOPK", "8"))


# ---------------------------------------------------------------------------
# Stages 3 + 4 + 5: ablation flags (LT 15-node DAG)
# ---------------------------------------------------------------------------
# The LT DAG has K=15 value-emitting nodes:
#     10 organ-age sub-tasks (cardiovascular, digestive, respiratory,
#                             endocrine/metabolic, nervous, hematologic,
#                             musculoskeletal/motor, urinary, immune,
#                             frailty),
#     4 LT-specific risk gradings (past medical history, rejection,
#                                  infection, postoperative complication),
#     1 integrative overall biological age.
# Plus one anchor sub-task (post-transplant pattern recognition,
# rationale-only).  The fused feature vector that feeds Cox is thus
# 15 (text) + 1 (VLM CT-derived risk) = 16 dimensions.
_FULL = {
    "use_role_layer":         True,
    "use_knowledge_layer":    True,
    "use_cot_protocol":       True,
    "use_output_contract":    True,

    "decomposition_k":        15,    # 1, 11, or 15
    "anchor_disease_pattern": True,

    "per_task_cot":           True,
    "n_samples":              1,

    # VLM branch: include the CT-derived risk in the fused feature vector.
    "use_vlm_branch":         True,
}


def _override(base, **kwargs):
    out = dict(base)
    out.update(kwargs)
    return out


ABLATION_PRESETS = {
    "full": _FULL,                                                          # A1.0

    # ---- A1: prompt-layer ablations -------------------------------------
    "A1.1": _override(_FULL, use_role_layer=False),
    "A1.2": _override(_FULL, use_knowledge_layer=False),
    "A1.3": _override(_FULL, use_cot_protocol=False, per_task_cot=False),
    "A1.4": _override(_FULL, use_output_contract=False),

    # ---- A2: decomposition-granularity ablations ------------------------
    "A2.1": _override(_FULL, decomposition_k=1),                            # overall age only
    "A2.2": _override(_FULL, decomposition_k=11),                           # organ + frailty + overall
    "A2.3": _override(_FULL, anchor_disease_pattern=False),                 # K=15 flat order

    # ---- A3: reasoning-strategy ablations -------------------------------
    "A3.1": _override(_FULL, per_task_cot=False),                           # zero-shot, no CoT

    # ---- A4: sampling-strategy ablations --------------------------------
    "A4.1": _override(_FULL, n_samples=1),                                  # T>0 single-shot
    "A4.2": _override(_FULL, n_samples=5),                                  # majority voting

    # ---- A6: modality ablation (LT-specific) ----------------------------
    "A6.1": _override(_FULL, use_vlm_branch=False),                         # text-only fusion
}

ACTIVE_PRESET = os.environ.get("LLM_MMI_PRESET", "full")
ABLATION      = ABLATION_PRESETS[ACTIVE_PRESET]

if ACTIVE_PRESET == "A4.1":
    TEMPERATURE = float(os.environ.get("LLM_MMI_TEMPERATURE", "0.7"))


# ---------------------------------------------------------------------------
# Multi-thread inference & I/O paths
# ---------------------------------------------------------------------------
# The LT cohorts are three single-centre transplant programmes; ``DATA_PATH``
# points at the manifest produced by ``LT/build_tree.py`` (one row per
# patient, with ``ID``, ``input`` = harmonised narrative, and a JSON column
# ``ct_keyframes`` listing the CAM-selected PNG paths).
THREAD_WORKERS    = int(os.environ.get("LT_WORKERS", "16"))
DATA_PATH         = os.environ.get(
    "LT_DATA",
    "/home/user/Agent/LT/yanzheng11-2.xlsx",
)
KEYFRAME_INDEX    = os.environ.get(
    "LT_KEYFRAME_INDEX",
    "/home/user/Agent/LT/site1.json",     # output of LT/build_tree.py + CAM
)
API_KEY_PATH      = os.environ.get(
    "LT_API_KEYS",
    "/home/user/Agent/LT/api.txt",
)
OUTPUT_PATH       = os.environ.get("LT_OUTPUT",  "lt_output.xlsx")
VLM_OUTPUT_PATH   = os.environ.get("LT_VLM_OUTPUT", "lt_vlm_output.xlsx")
FUSION_OUTPUT     = os.environ.get("LT_FUSION_OUTPUT", "lt_fusion.xlsx")
