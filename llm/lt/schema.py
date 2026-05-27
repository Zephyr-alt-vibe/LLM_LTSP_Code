"""15-node DAG output schemas for LLM-MMI-LT (Stage 3).

The DAG ``G = (T, E)`` consists of:

* ``tau_0``           — *post-transplant pattern recognition* (anchor,
  rationale only — not counted in the 15 value-emitting nodes);
* ``tau_1..tau_10``   — 9 organ-system biological ages + frailty age;
* ``tau_11..tau_14``  — 4 LT-specific postoperative risk gradings
  (past medical history, rejection, infection, postoperative complication);
* ``tau_15``          — integrative *overall biological age*.

This module emits the JSON schema fragment that is dropped into the
*output-contract* layer of the LT prompt (see
:func:`llm_mmi.lt.prompt.assemble_prompt`).  The shape mirrors the
multimorbidity package (:mod:`llm_mmi.schema`) so that downstream parsers
and Cox-fitting code remain compatible.
"""

import random


# ----- Node lists, ordered by topology ------------------------------------
ANCHOR_NODE = ("inference process disease",
               "string",
               "Please give your inference process of the current post-"
               "transplant clinical pattern, integrating CT reports, "
               "biochemistry trends, and discharge summary findings.")

ORGAN_NODES = [
    ("cardiovascular system age",        "int",
        "predicting the cardiovascular system age post-transplantation"),
    ("digestive system age",             "int",
        "predicting the digestive system age post-transplantation"),
    ("respiratory system age",           "int",
        "predicting the respiratory system age (graft-bearing) post-"
        "transplantation"),
    ("endocrine/metabolic system age",   "int",
        "predicting the endocrine/metabolic system age post-"
        "transplantation"),
    ("nervous system age",               "int",
        "predicting the nervous system age post-transplantation"),
    ("hematologic system age",           "int",
        "predicting the hematologic system age post-transplantation"),
    ("musculoskeletal/motor system age", "int",
        "predicting the musculoskeletal/motor system age post-"
        "transplantation"),
    ("urinary system age",               "int",
        "predicting the urinary system age post-transplantation"),
    ("immune system age",                "int",
        "predicting the immune system age post-transplantation (under "
        "immunosuppression)"),
    ("frailty age",                      "int",
        "predicting the frailty age post-transplantation"),
]

# Four LT-specific postoperative risk gradings replace the multimorbidity
# package's six external-factor gradings.  These dimensions are the core
# clinical risk axes for the early-to-mid post-transplant period.
LT_RISK_NODES = [
    ("medical history grading",                 "grade",
        "grading the contribution of past medical history to "
        "post-transplant prognosis"),
    ("rejection risk grading",                  "grade",
        "grading the risk of acute or chronic graft rejection in the "
        "context of the current clinical pattern"),
    ("infection risk grading",                  "grade",
        "grading the risk of post-transplant infection (bacterial, "
        "viral, fungal) under the current immunosuppression"),
    ("postoperative complication risk grading", "grade",
        "grading the risk of postoperative complications (e.g. "
        "anastomotic, haemodynamic, renal) given the perioperative "
        "course"),
]

INTEGRATIVE_NODE = ("overall biological age", "int",
                    "predicting the overall post-transplant biological age "
                    "by integrating all organ-specific ages and risk "
                    "gradings")

GRADING_LEVELS = [
    "high-risk", "medium-high-risk", "medium-risk",
    "medium-low-risk", "low-risk",
]


# Numeric coding of the ordinal grades.  Higher value -> higher hazard.
GRADING_ENCODING = {
    "low-risk":         1,
    "medium-low-risk":  2,
    "medium-risk":      3,
    "medium-high-risk": 4,
    "high-risk":        5,
}


# ----- Public API ----------------------------------------------------------
def select_nodes(decomposition_k=15, anchor_disease_pattern=True, seed=0):
    """Return the ordered list of (key, type, rationale-blurb) triples that
    constitute the schema for a given ablation configuration."""

    if decomposition_k == 1:
        return [INTEGRATIVE_NODE]

    if decomposition_k == 11:
        body = list(ORGAN_NODES) + [INTEGRATIVE_NODE]
    elif decomposition_k == 15:
        body = list(ORGAN_NODES) + list(LT_RISK_NODES) + [INTEGRATIVE_NODE]
    else:
        raise ValueError(
            f"decomposition_k must be 1, 11, or 15; got {decomposition_k!r}"
        )

    if anchor_disease_pattern:
        return [ANCHOR_NODE] + body

    rng = random.Random(seed)
    rng.shuffle(body)
    return body


def render_json_schema(decomposition_k=15,
                       anchor_disease_pattern=True,
                       per_task_cot=True,
                       seed=0):
    """Render the JSON schema fragment that goes inside ``\`\`\`json … \`\`\``.

    Matches the layout convention in :mod:`llm_mmi.schema` so that the same
    parser in :mod:`llm_mmi.lt.fusion` can recover the value vector.
    """

    nodes = select_nodes(decomposition_k=decomposition_k,
                         anchor_disease_pattern=anchor_disease_pattern,
                         seed=seed)

    grade_choices = '", "'.join(GRADING_LEVELS)

    counter = 0
    lines = ["{"]
    for key, typ, blurb in nodes:
        is_anchor = (key == ANCHOR_NODE[0])

        if per_task_cot:
            if is_anchor:
                lines.append(f'  "inference process disease": string,')
                lines.append(f'  // {blurb}')
            else:
                counter += 1
                lines.append(f'  "inference process {counter}": string,')
                if typ == "grade":
                    lines.append(
                        f'  // Please give your inference process of {blurb}. '
                        f'One of "{grade_choices}" should be assigned.'
                    )
                else:
                    lines.append(
                        f'  // Please give your inference process of {blurb}.'
                    )

        if is_anchor:
            lines.append("")
            continue

        if typ == "int":
            lines.append(f'  "{key}": int,')
        else:  # grade
            lines.append(f'  "{key}": string,')
        lines.append("")

    while lines and lines[-1] == "":
        lines.pop()
    if lines[-1].rstrip().endswith(","):
        lines[-1] = lines[-1].rstrip().rstrip(",")
    lines.append("}")
    return "\n".join(lines)


# ----- VARS_USE (for downstream fusion / Cox) -----------------------------
# Field order matches the columns expected by :mod:`llm_mmi.lt.fusion` and
# the Cox-fitting helper.  The VLM branch contributes a single binary
# variable appended at position 16.
ORGAN_AGE_FIELDS = [k for (k, _, _) in ORGAN_NODES]
LT_RISK_FIELDS   = [k for (k, _, _) in LT_RISK_NODES]
INTEGRATIVE_FIELD = INTEGRATIVE_NODE[0]
VLM_FIELD = "ct-derived risk"

# 15-D text vector then +1 CT-derived risk -> 16-D fused vector.
TEXT_VARS_USE = ORGAN_AGE_FIELDS + LT_RISK_FIELDS + [INTEGRATIVE_FIELD]
FUSED_VARS_USE = TEXT_VARS_USE + [VLM_FIELD]
