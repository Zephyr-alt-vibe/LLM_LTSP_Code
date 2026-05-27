"""Four-layer prompt assembly for LLM-MMI (Method.md, section "B.5").

The full prompt is

    P(n_i) = R || K || S_cot || S_out || n_i

where

* ``R``      — *Role* layer, declares the systems-medicine multimorbidity
  expert persona;
* ``K``      — *Knowledge* layer, injects the network-coupling prior, the
  24-marker sentinel-indicator list, and the evidence-weighting rule;
* ``S_cot``  — *Reasoning protocol* layer, defines the typed CoT scaffold
  (abnormality scan -> coupling analysis -> interaction-conditional grading);
* ``S_out``  — *Output contract* layer, the task instruction + JSON schema;
* ``n_i``    — the harmonised English health narrative.

Each layer is independently switchable through the ablation flags consumed
from :mod:`llm_mmi.config` so that experiments A1.1–A1.4 of the paper can
be reproduced by toggling exactly one boolean.
"""

from . import config
from . import schema


# ---------------------------------------------------------------------------
# Layer R: role specification
# ---------------------------------------------------------------------------
ROLE_LAYER = (
    "Your role: You are a systems medicine scholar specializing in "
    "multimorbidity and disease-disease interactions. You are knowledgeable "
    "about the normal ranges of various laboratory test results and what "
    "abnormal values represent. You can integrate clinical data and "
    "laboratory results, assess the synergistic/antagonistic effects between "
    "diseases, and map them to overall biological age, the biological age of "
    "specific systems, and the grading of external factors. If the relevant "
    "indicators are currently healthy, the biological age of each system "
    "should be reasonably reduced."
)


# ---------------------------------------------------------------------------
# Layer K: domain-knowledge module
# ---------------------------------------------------------------------------
KNOWLEDGE_LAYER = (
    "Background Knowledge: In the context of multimorbidity, individual risk "
    "arises from the network coupling between diseases, organs, and systems, "
    "rather than simply the additive risk of individual diseases. Multiple "
    "shared pathways (e.g., metabolism, inflammation, "
    "endothelial/microcirculation, neurohumoral, immune) can form cross-loops "
    "and cascading feedback loops, such that an imbalance in one system "
    "amplifies the burden on another system, resulting in synergistic damage. "
    "Conversely, antagonistic effects may occur due to pathway compensation "
    "or therapeutic effects. The same disease may exhibit effect modification "
    "under different comorbidity combinations, control levels, and disease "
    "stages. When compensation is depleted or key physiological thresholds "
    "are triggered, system load may undergo a nonlinear transition, "
    "accelerating the consumption of health reserves. In addition to "
    "intrinsic disease-organ interactions, external factors impose "
    "structural impacts on the eight major organ systems and multimorbidity "
    "patterns. Psychological stress, diet quality, unhealthy behaviors and "
    "habits (such as smoking, alcohol consumption, lack of exercise), "
    "socioeconomic resources (such as income), and genetic susceptibility "
    "(family history) can selectively exacerbate or alleviate the burden on "
    "specific systems, reshape shared pathways such as metabolism, "
    "inflammation, or neurohumoral axes, and modulate how comorbid diseases "
    "cluster and interact, ultimately influencing the evolution of the "
    "biological age of specific systems and the final overall biological "
    "age. When lethal diseases and associated indicators appear "
    "simultaneously, the relevant system's age and risk grade should be "
    "elevated. Particular attention should be given to indicators such as "
    "lactate, advanced cancer, hypertensive crisis, C-reactive protein, "
    "GlycA, high Cystatin C, high RDW, low albumin/weight loss, extremely "
    "low vitamin D, high Lp(a) + vascular leakage, extremely low "
    "neutrophils/high NLR, high GGT + high ketones, recurrent falls, "
    "Fed-up + insomnia, tooth loss, pulse pressure difference > 60 mmHg, "
    "diagnoses of more than three new diseases in the past year, "
    "monocytes > 10%, extremely low urine creatinine, normal HbA1c but "
    "random blood glucose spikes, increased MPV, extremely wide pulse "
    "pressure difference, and branched-chain amino acids (BCAAs) depletion. "
    "Healthy lifestyle behaviors can appropriately reduce the predicted "
    "organ age, such as not smoking in the past and currently, and not "
    "engaging in excessive alcohol consumption in the past and present. "
    "Some laboratory test indicators within the normal range can also "
    "appropriately reduce the predicted organ age."
)


# ---------------------------------------------------------------------------
# Layer S_cot: reasoning protocol
# ---------------------------------------------------------------------------
COT_PROTOCOL_LAYER = (
    "Analysis Logic and Chain of Thought Recommendation (Key Steps): "
    "Abnormal value weighting (key step): Scan laboratory data. Note: The "
    "weight of objective indicators is always higher than that of subjective "
    "self-reports."
)


# ---------------------------------------------------------------------------
# Layer S_out: output contract (task instruction + JSON schema)
# ---------------------------------------------------------------------------
_TASK_INSTRUCTION_FULL = (
    "Your Task: Based on the individual health information provided below, "
    "predict the comorbidity-related health condition of the person. You "
    "should first provide your reasoning process, clearly describing how "
    "the current multimorbidity pattern, laboratory findings, and external "
    "factors jointly affect system load and interactions. Then, provide the "
    "biological age of the following systems: cardiovascular system, "
    "digestive system, respiratory system, endocrine/metabolic system, "
    "nervous system, hematologic system, musculoskeletal/movement system, "
    "urinary system, immune system, and frailty age. Additionally, provide "
    "the grading for psychological health, dietary health, behavioral/habit "
    "health, medical history, income, and family history. For each of these "
    "dimensions, you must assign one of five categories: high-risk, "
    "medium-high-risk, medium-risk, medium-low-risk, or low-risk, and all "
    "grades must be determined in conjunction with the patient's specific "
    "comorbidity characteristics (i.e., the same external exposure should "
    "not receive the same risk level under different comorbidity "
    "conditions). Finally, based on the inferred biological age of specific "
    "systems and the graded external factors, integrate all information to "
    "derive and report the overall biological age."
)

_TASK_INSTRUCTION_K10 = (
    "Your Task: Based on the individual health information provided below, "
    "predict the biological age of each organ system (cardiovascular, "
    "digestive, respiratory, endocrine/metabolic, nervous, hematologic, "
    "musculoskeletal/motor, urinary, immune) plus a frailty age, and "
    "integrate them into an overall biological age. External grading "
    "dimensions are intentionally omitted in this configuration."
)

_TASK_INSTRUCTION_K1 = (
    "Your Task: Based on the individual health information provided below, "
    "predict a single overall biological age for the person."
)

_FORMAT_PREAMBLE = (
    "The format of your answer is JSON, please do not give any additional "
    "output, please refer to the following format to give your answer:"
)

_TASK_INSTRUCTION_FREEFORM = (
    "Your Task: Based on the individual health information provided below, "
    "predict the comorbidity-related health condition of the person. "
    "Provide a free-form clinical assessment in plain English."
)


def _select_task_instruction(decomposition_k):
    if decomposition_k == 1:
        return _TASK_INSTRUCTION_K1
    if decomposition_k == 10:
        return _TASK_INSTRUCTION_K10
    return _TASK_INSTRUCTION_FULL


def output_contract_layer(decomposition_k=17,
                          anchor_disease_pattern=True,
                          per_task_cot=True,
                          seed=0):
    """Render the *Output contract* layer ``S_out``."""

    instruction = _select_task_instruction(decomposition_k)
    json_schema = schema.render_json_schema(
        decomposition_k=decomposition_k,
        anchor_disease_pattern=anchor_disease_pattern,
        per_task_cot=per_task_cot,
        seed=seed,
    )
    return f"{instruction}\n{_FORMAT_PREAMBLE}\n```json\n{json_schema}\n```"


# ---------------------------------------------------------------------------
# Final assembly
# ---------------------------------------------------------------------------
_NARRATIVE_TAIL = (
    "Give you the person's health information as:\n"
    "{narrative}\n"
    "Based on the information above, your answer is:"
)


def assemble_prompt(narrative, ablation=None, seed=0):
    """Build the full prompt ``P(n_i)`` for one participant.

    ``ablation`` defaults to :data:`llm_mmi.config.ABLATION` so that the
    default invocation reproduces the paper's main configuration.
    """

    if ablation is None:
        ablation = config.ABLATION

    parts = []

    if ablation["use_role_layer"]:
        parts.append(ROLE_LAYER)
    if ablation["use_knowledge_layer"]:
        parts.append(KNOWLEDGE_LAYER)
    if ablation["use_cot_protocol"]:
        parts.append(COT_PROTOCOL_LAYER)

    if ablation["use_output_contract"]:
        parts.append(output_contract_layer(
            decomposition_k=ablation["decomposition_k"],
            anchor_disease_pattern=ablation["anchor_disease_pattern"],
            per_task_cot=ablation["per_task_cot"],
            seed=seed,
        ))
    else:
        # A1.4: free-form clinical assessment, parsed post-hoc.
        parts.append(_TASK_INSTRUCTION_FREEFORM)

    parts.append(_NARRATIVE_TAIL.format(narrative=narrative))
    return "\n\n".join(parts)
