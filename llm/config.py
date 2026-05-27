"""LLM-MMI inference configuration.

Stage-wise mapping (see Method.md, section "B. Methods"):
    Stage 2 — frozen policy model      -> POLICY_BACKBONES, ACTIVE_BACKBONE
    Stage 3 — DAG decomposition        -> ABLATION["decomposition_k"]
    Stage 4 — four-layer prompt schema -> ABLATION["use_*_layer"]
    Stage 5 — deterministic sampling   -> TEMPERATURE, ABLATION["n_samples"]
    Stage 6 — aggregation Phi + Cox    -> handled in eval/*.R

Ablation presets (see Method.md, section "C.6") can be selected through the
``LLM_MMI_PRESET`` environment variable, e.g.

    LLM_MMI_PRESET=A1.2 python -m llm_mmi.inference

Backbone witnesses for the cross-backbone portability experiment (Method.md
section "C.7") can be selected through ``LLM_MMI_BACKBONE``.
"""

import os


# ---------------------------------------------------------------------------
# Stage 2: frozen policy backbones
# ---------------------------------------------------------------------------
# All five backbones are accessed through an OpenAI-compatible interface;
# downstream gateways (e.g. apiyi) often re-route the same endpoint.  The
# entries below are the *direct* endpoints; users with an aggregator may
# point every entry at the aggregator and update only ``model``.
POLICY_BACKBONES = {
    "deepseek-v3": {
        "model":    "deepseek-chat",
        "base_url": "https://api.deepseek.com/v1",
    },
    "gpt-5-nano": {
        "model":    "gpt-5-nano",
        "base_url": "https://api.openai.com/v1",
    },
    "grok-4-fast": {
        "model":    "grok-4-fast",
        "base_url": "https://api.x.ai/v1",
    },
    "llama-3.3-70b": {
        "model":    "llama-3.3-70b-instruct",
        "base_url": "https://api.together.xyz/v1",
    },
    "gemini-2.5-flash": {
        "model":    "gemini-2.5-flash-preview-05-20",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
    },
    # The aggregator endpoint actually used in the production runs.  Keep it
    # available as a non-default option so existing users do not need to
    # migrate their API keys.
    "apiyi-aggregator": {
        "model":    "qwen3.5-397b-a17b",
        "base_url": "https://api.apiyi.com/v1",
    },
}

ACTIVE_BACKBONE = os.environ.get("LLM_MMI_BACKBONE", "deepseek-v3")


# ---------------------------------------------------------------------------
# Stage 5: deterministic single-pass sampling
# ---------------------------------------------------------------------------
TEMPERATURE     = 0.0
TIMEOUT_SECONDS = 600


# ---------------------------------------------------------------------------
# Stages 3 + 4 + 5: ablation flags
# ---------------------------------------------------------------------------
# The "full" preset reproduces LLM-MMI A1.0; every other preset toggles a
# single component so each experiment maps to exactly one paper claim.
_FULL = {
    # A1.x: prompt-layer ablations
    "use_role_layer":         True,
    "use_knowledge_layer":    True,
    "use_cot_protocol":       True,
    "use_output_contract":    True,

    # A2.x: decomposition granularity
    "decomposition_k":        17,    # 1, 10, or 17
    "anchor_disease_pattern": True,  # tau_0 placed first as shared context

    # A3.x: per-task chain-of-thought rationale
    "per_task_cot":           True,

    # A4.x: stochastic test-time scaling
    "n_samples":              1,     # >1 enables majority voting
}


def _override(base, **kwargs):
    out = dict(base)
    out.update(kwargs)
    return out


ABLATION_PRESETS = {
    "full": _FULL,                                                          # A1.0

    # ---- A1: prompt-layer ablations -------------------------------------
    "A1.1": _override(_FULL, use_role_layer=False),                         # drop Role
    "A1.2": _override(_FULL, use_knowledge_layer=False),                    # drop Knowledge
    "A1.3": _override(_FULL, use_cot_protocol=False, per_task_cot=False),   # drop CoT protocol
    "A1.4": _override(_FULL, use_output_contract=False),                    # drop Output contract

    # ---- A2: decomposition-granularity ablations ------------------------
    "A2.1": _override(_FULL, decomposition_k=1),                            # K = 1 (overall age only)
    "A2.2": _override(_FULL, decomposition_k=10),                           # K = 10 (organ + frailty)
    "A2.3": _override(_FULL, anchor_disease_pattern=False),                 # K = 17 flat order

    # ---- A3: reasoning-strategy ablations -------------------------------
    "A3.1": _override(_FULL, per_task_cot=False),                           # zero-shot, no CoT
    # A3.2 (per-task CoT, no anchor) is operationally identical to A2.3;
    # use A2.3 to reproduce it.

    # ---- A4: sampling-strategy ablations --------------------------------
    "A4.1": _override(_FULL, n_samples=1),                                  # stochastic single-shot
    # NOTE: A4.1 also requires TEMPERATURE=0.7 — set via env var.
    "A4.2": _override(_FULL, n_samples=5),                                  # majority voting, N = 5
}

# Active preset selectable via env var (default = full LLM-MMI).
ACTIVE_PRESET = os.environ.get("LLM_MMI_PRESET", "full")
ABLATION      = ABLATION_PRESETS[ACTIVE_PRESET]

# A4.1 specifically requires breaking determinism; honour env-supplied T.
if ACTIVE_PRESET == "A4.1":
    TEMPERATURE = float(os.environ.get("LLM_MMI_TEMPERATURE", "0.7"))


# ---------------------------------------------------------------------------
# Multi-thread inference & I/O paths
# ---------------------------------------------------------------------------
THREAD_WORKERS = int(os.environ.get("LLM_MMI_WORKERS", "50"))
DATA_PATH      = os.environ.get(
    "LLM_MMI_DATA",
    r"/Users/zj004/Desktop/测试/UKB/ukb.all.xlsx",
)
API_KEY_PATH   = os.environ.get(
    "LLM_MMI_API_KEYS",
    r"/Users/zj004/Desktop/测试/api.txt",
)
OUTPUT_PATH    = os.environ.get("LLM_MMI_OUTPUT", "output.xlsx")
