"""Four-layer prompt assembly for LLM-MMI-LT (Stage 4, text branch).

The full prompt is

    P(n_i) = R || K || S_cot || S_out || n_i

where each layer is the LT-specific instantiation of its multimorbidity
counterpart (:mod:`llm_mmi.prompt`).  The content matches the prompt
template in ``LT/baseline.md`` §1.4 (eMethods 1).
"""

from . import config
from . import schema


# ---------------------------------------------------------------------------
# Layer R: role specification (lung-transplant clinical expert)
# ---------------------------------------------------------------------------
ROLE_LAYER = (
    "Your role: You are a clinical expert specializing in lung "
    "transplantation and critical care medicine, with a profound "
    "understanding of long-term post-transplant survival, complication "
    "risk assessment, biomarkers, and clinical manifestations during the "
    "postoperative recovery period. You are adept at integrating multiple "
    "sources of medical data, including CT reports, biochemical test "
    "results, and discharge summaries. You can synthesize imaging data, "
    "laboratory findings, and clinical histories of lung transplant "
    "recipients to evaluate postoperative risk and predict long-term "
    "survival and functional recovery. You are particularly skilled in "
    "analyzing changes in radiological features, laboratory parameters, "
    "and clinical events occurring during hospitalization, in conjunction "
    "with patients' biological age, immune status, and other system "
    "functions, to provide physicians with personalized prognostic "
    "predictions."
)


# ---------------------------------------------------------------------------
# Layer K: graft-host coupling and iatrogenic-factor knowledge
# ---------------------------------------------------------------------------
KNOWLEDGE_LAYER = (
    "Background Knowledge: In the complex context of lung transplantation, "
    "individual risk arises from the dynamic coupling between the graft, "
    "the host immune system, and systemic organ function, rather than "
    "simply the additive risk of isolated rejection or infection. Multiple "
    "shared pathways — such as immune regulation, inflammatory cascades, "
    "hemodynamics, endothelial/microcirculatory function, and drug "
    "metabolism — can form intersecting loops and cascade feedback "
    "circuits, whereby imbalance in one system (e.g., latent infection "
    "activation due to immunosuppression) can exacerbate the burden on "
    "another (e.g., triggering systemic inflammatory responses that "
    "impair graft or renal function), resulting in synergistic injury. "
    "Conversely, the establishment of immune tolerance or effective "
    "cardiopulmonary compensatory mechanisms may confer protective "
    "effects. The same clinical parameter change (e.g., hypoxemia) may "
    "exert different effect modifications depending on the type of "
    "complication (e.g., immune rejection versus infection), drug "
    "concentration control (e.g., fluctuations in FK506 levels), and "
    "stage of the clinical course. When physiological compensation is "
    "exhausted or critical physiological thresholds are reached (e.g., "
    "lactate accumulation, oxygenation index inflection points, or renal "
    "deterioration), system load may undergo nonlinear transitions, "
    "accelerating depletion of physiological reserves and precipitating "
    "multi-organ dysfunction. Beyond intrinsic graft–host interactions, "
    "iatrogenic factors — such as immunosuppressive regimens, anti-"
    "infective strategies, donor organ quality, and perioperative "
    "supportive care — also profoundly influence postoperative outcomes. "
    "These factors can selectively modulate the burden on specific organ "
    "systems, reshape shared pathways including immune and coagulation "
    "circuits, thereby affecting the interplay between rejection and "
    "infection, and ultimately determining long-term graft survival and "
    "overall patient prognosis."
)


# ---------------------------------------------------------------------------
# Layer S_cot: reasoning protocol with evidence hierarchy
# ---------------------------------------------------------------------------
COT_PROTOCOL_LAYER = (
    "Analytical Logic and Chain-of-Thought Recommendations (Key Steps): "
    "Outlier Weighting (Key Steps): Scan time-sequential CT images, CT "
    "textual reports, biochemical test reports, and discharge summaries. "
    "Note: During the reasoning process, please adhere to the following "
    "hierarchy of evidence weight: Confirmatory textual report evidence "
    "> Dynamic imaging evolution > Trends in biochemical indicator "
    "changes. The same external exposures should not be assigned the "
    "same risk level under different transplant conditions."
)


# ---------------------------------------------------------------------------
# Layer S_out: output contract (task instruction + JSON schema)
# ---------------------------------------------------------------------------
_TASK_INSTRUCTION_FULL = (
    "Your Task: Based on the provided preoperative, postoperative, and "
    "post-discharge clinical data of lung transplant patients — including "
    "CT imaging reports, biochemical test results, and discharge "
    "summaries — predict the patient's postoperative prognosis. You "
    "should first present the reasoning process, clearly describing how "
    "to comprehensively assess the patient's postoperative recovery using "
    "the current clinical information. Next, provide the biological age "
    "for the following systems: cardiovascular, digestive, respiratory, "
    "endocrine/metabolic, nervous, hematologic, musculoskeletal/motor, "
    "urinary, immune system, and frailty age. Then, predict the following "
    "clinical dimensions: medical history grading, rejection risk "
    "grading, infection risk grading, and postoperative complication "
    "risk grading. For each dimension, assign one of five categories: "
    "high-risk, medium-high-risk, medium-risk, medium-low-risk, or "
    "low-risk. All gradings should be determined in the context of the "
    "patient's specific lung transplant characteristics (i.e., the same "
    "external exposures should not be assigned the same risk level under "
    "different transplant conditions). Finally, based on the inferred "
    "biological ages of each system and the graded postoperative risk "
    "factors, integrate all information to derive and report the overall "
    "biological age."
)

_TASK_INSTRUCTION_K11 = (
    "Your Task: Based on the individual clinical information provided "
    "below, predict the biological age of each organ system "
    "(cardiovascular, digestive, respiratory, endocrine/metabolic, "
    "nervous, hematologic, musculoskeletal/motor, urinary, immune) plus "
    "a frailty age, and integrate them into an overall biological age. "
    "Postoperative risk gradings are intentionally omitted in this "
    "configuration."
)

_TASK_INSTRUCTION_K1 = (
    "Your Task: Based on the individual clinical information provided "
    "below, predict a single overall post-transplant biological age for "
    "the patient."
)

_FORMAT_PREAMBLE = (
    "The format of your answer is JSON, please do not give any "
    "additional output, please refer to the following format to give "
    "your answer:"
)

_TASK_INSTRUCTION_FREEFORM = (
    "Your Task: Based on the individual clinical information provided "
    "below, predict the post-transplant prognosis of the patient in "
    "free-form plain English."
)


def _select_task_instruction(decomposition_k):
    if decomposition_k == 1:
        return _TASK_INSTRUCTION_K1
    if decomposition_k == 11:
        return _TASK_INSTRUCTION_K11
    return _TASK_INSTRUCTION_FULL


def output_contract_layer(decomposition_k=15,
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
    "Give you the patient's longitudinal clinical narrative as:\n"
    "{narrative}\n"
    "Based on the information above, your answer is:"
)


def assemble_prompt(narrative, ablation=None, seed=0):
    """Build the full text-branch prompt ``P(n_i)`` for one patient.

    ``ablation`` defaults to :data:`llm_mmi.lt.config.ABLATION` so that the
    default invocation reproduces the full LT-LLM-MMI configuration.
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
        parts.append(_TASK_INSTRUCTION_FREEFORM)

    parts.append(_NARRATIVE_TAIL.format(narrative=narrative))
    return "\n\n".join(parts)
