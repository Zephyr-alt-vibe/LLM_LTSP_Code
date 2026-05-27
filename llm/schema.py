"""18-node DAG output schemas (Method.md, section "B.4").

The DAG ``G = (T, E)`` consists of:

* ``tau_0``  — *multimorbidity pattern recognition* (anchor node);
* ``tau_1..tau_10``  — 9 organ-system biological ages + frailty age;
* ``tau_11..tau_16`` — 6 external grading dimensions;
* ``tau_17`` — integrative *overall biological age*.

This module emits the JSON schema fragment that is dropped into the
*output-contract* layer of the prompt (see ``prompt.OUTPUT_CONTRACT_LAYER``).
The schema is parameterised by

* ``decomposition_k``       — 1 / 10 / 17 (Method.md A2.1 / A2.2 / A2.3+);
* ``anchor_disease_pattern`` — keep ``tau_0`` (Method.md A1.0) or drop and
  randomise the order (Method.md A2.3);
* ``per_task_cot``          — embed an ``inference process N`` rationale per
  node (Method.md A1.0) or drop all rationales (Method.md A3.1).
"""

import random


# ----- Node lists, ordered by paper-level topology ------------------------
ANCHOR_NODE = ("inference process disease",
               "string",
               "Please give your inference process of the current "
               "multimorbidity pattern.")

ORGAN_NODES = [
    ("cardiovascular system age",        "int",
        "predicting the cardiovascular system age"),
    ("digestive system age",             "int",
        "predicting the digestive system age"),
    ("respiratory system age",           "int",
        "predicting the respiratory system age"),
    ("endocrine/metabolic system age",   "int",
        "predicting the endocrine/metabolic system age"),
    ("nervous system age",               "int",
        "predicting the nervous system age"),
    ("hematologic system age",           "int",
        "predicting the hematologic system age"),
    ("musculoskeletal/motor system age", "int",
        "predicting the musculoskeletal/motor system age"),
    ("urinary system age",               "int",
        "predicting the urinary system age"),
    ("immune system age",                "int",
        "predicting the immune system age"),
    ("frailty age",                      "int",
        "predicting the frailty age"),
]

EXTERNAL_NODES = [
    ("psychological health grading",   "grade",
        "grading psychological health in the context of the current "
        "multimorbidity profile"),
    ("dietary health grading",         "grade",
        "grading dietary health in the context of the current "
        "multimorbidity profile"),
    ("behaviors/habits health grading", "grade",
        "grading unhealthy behaviors/habits health in the context of the "
        "current multimorbidity profile"),
    ("income grading",                 "grade",
        "grading income-related health impact in the context of the "
        "current multimorbidity profile"),
    ("family history grading",         "grade",
        "grading family history in the context of the current "
        "multimorbidity profile"),
    ("past medical history grading",   "grade",
        "grading past medical history in the context of the current "
        "multimorbidity profile"),
]

INTEGRATIVE_NODE = ("overall biological age", "int",
                    "predicting the overall biological age by all information")

GRADING_LEVELS = [
    "high-risk", "medium-high-risk", "medium-risk",
    "medium-low-risk", "low-risk",
]


# ----- Public API ----------------------------------------------------------
def select_nodes(decomposition_k=17, anchor_disease_pattern=True, seed=0):
    """Return the ordered list of (key, type, rationale-blurb) triples that
    constitute the schema for a given ablation configuration."""

    if decomposition_k == 1:
        return [INTEGRATIVE_NODE]

    if decomposition_k == 10:
        body = list(ORGAN_NODES) + [INTEGRATIVE_NODE]
    elif decomposition_k == 17:
        body = list(ORGAN_NODES) + list(EXTERNAL_NODES) + [INTEGRATIVE_NODE]
    else:
        raise ValueError(
            f"decomposition_k must be 1, 10, or 17; got {decomposition_k!r}"
        )

    if anchor_disease_pattern:
        return [ANCHOR_NODE] + body

    # A2.3 / A3.2: drop tau_0 and randomise the order to test whether the
    # DAG ordering itself contributes (rather than merely the node set).
    rng = random.Random(seed)
    rng.shuffle(body)
    return body


def render_json_schema(decomposition_k=17,
                       anchor_disease_pattern=True,
                       per_task_cot=True,
                       seed=0):
    """Render the JSON schema fragment that goes inside ``\`\`\`json … \`\`\``.

    Behaviour matches the original Prompts.py output verbatim under
    ``decomposition_k=17, anchor_disease_pattern=True, per_task_cot=True``.
    """

    nodes = select_nodes(decomposition_k=decomposition_k,
                         anchor_disease_pattern=anchor_disease_pattern,
                         seed=seed)

    grade_choices = '", "'.join(GRADING_LEVELS)

    # Numbering for "inference process N" follows the order of the schema,
    # excluding the dedicated anchor field which is always called
    # "inference process disease".
    counter = 0
    lines = ["{"]
    for idx, (key, typ, blurb) in enumerate(nodes):
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

        # Skip the value field for the anchor — it is intentionally CoT-only.
        if is_anchor:
            lines.append("")
            continue

        if typ == "int":
            lines.append(f'  "{key}": int,')
        else:  # grade
            lines.append(f'  "{key}": string,')
        lines.append("")

    # Strip trailing blank line and trailing comma on the last value field
    # to keep the schema valid JSON-ish.
    while lines and lines[-1] == "":
        lines.pop()
    if lines[-1].rstrip().endswith(","):
        lines[-1] = lines[-1].rstrip().rstrip(",")
    lines.append("}")
    return "\n".join(lines)
