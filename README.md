# LLM-LTSP

**LLM-LTSP** (*Large Language Model for Lung Transplant Survival Prediction*) is a six-stage frozen-policy multimodal reasoning pipeline for post-lung-transplant survival prediction. It converts longitudinal clinical text and thoracic CT keyframes into a 16-dimensional fused representation: 15 text-derived clinical dimensions plus one CT-derived binary risk variable. The fused vector is designed for Cox proportional-hazards survival modelling at 12, 18, and 24 months after lung transplantation.

This README maps each methodological stage to the implementing code and gives reproducible entry points for default inference, ablations, and downstream Cox fusion.

---

## 1. Repository layout

```text
.
├── LICENSE.txt
├── dataset_example/
│   ├── README.md                   # input schema notes for the example spreadsheet
│   └── example_input.xlsx          # small example input file for inference smoke tests
├── llm/                            # Python package namespace used by this upload
│   ├── __init__.py                 # base framework namespace; LTSP lives in llm.lt
│   ├── config.py                   # legacy base-framework configuration
│   ├── schema.py                   # legacy base-framework schema
│   ├── prompt.py                   # legacy base-framework prompt assembly
│   ├── inference.py                # legacy base-framework inference runner
│   ├── lt/                         # LLM-LTSP implementation
│   │   ├── __init__.py
│   │   ├── config.py               # LTSP backbones, ablation presets, I/O paths
│   │   ├── schema.py               # 15-node text DAG and 16-D fused feature list
│   │   ├── prompt.py               # four-layer text-branch prompt assembly
│   │   ├── vlm.py                  # CT-keyframe VLM branch and risk parser
│   │   ├── inference.py            # population-scale text + VLM inference runner
│   │   ├── fusion.py               # text/VLM parsing, 16-D fusion, Cox PH metrics
│   │   └── imaging.py              # wrapper/documentation for upstream CT preprocessing
│   └── train/                      # optional outcome-grounded SFT/RFT utilities
└── README.md
```

The GitHub entry point for the LTSP model is `llm/lt/`. The top-level `llm/config.py`, `llm/prompt.py`, `llm/schema.py`, and `llm/inference.py` are retained from the base framework for compatibility, but the lung-transplant survival pipeline should be run through `llm.lt`.

---

## 2. Methodology → code mapping

| Stage | Method anchor | Implementation | Main knobs |
|---|---|---|---|
| 1a. Clinical-text harmonisation `g_txt(·)` | clinical-text harmonisation | external preprocessing; the Excel `input` column is the harmonised English narrative | `LT_DATA` |
| 1b. CT harmonisation `g_img(·)` | CT harmonisation | upstream DICOM → PNG → CAM/keyframe manifest; wrapper in `llm/lt/imaging.py::run_preprocessing()` | `LT_KEYFRAME_INDEX`, `LT_VLM_TOPK` |
| 2. Frozen policy backbone `π_θ` | policy backbone selection | `llm/lt/config.py::POLICY_BACKBONES` | `LT_BACKBONE`, `LT_VLM_BACKBONE`, `LT_INTERNVL_BASE_URL` |
| 3. Hierarchical decomposition | task decomposition | `llm/lt/schema.py::render_json_schema()` for the 15-value text DAG; `llm/lt/vlm.py::VLM_PROMPT` for the 4-step CT DAG | `decomposition_k`, `anchor_disease_pattern`, `use_vlm_branch` |
| 4. Four-layer prompt `P = R ∥ K ∥ S_cot ∥ S_out ∥ n` | prompt construction | `llm/lt/prompt.py::assemble_prompt()` and `llm/lt/vlm.py::build_vlm_messages()` | `use_role_layer`, `use_knowledge_layer`, `use_cot_protocol`, `use_output_contract` |
| 5. Deterministic single-pass sampling | deterministic decoding | `llm/lt/inference.py::_process_row()`; text and VLM branches use greedy decoding by default | `LLM_MMI_PRESET=A4.*`, `LLM_MMI_TEMPERATURE` |
| 6. Multimodal fusion + Cox PH risk mapping | multimodal fusion and survival modelling | `llm/lt/fusion.py::run_fusion_pipeline()` | `LT_HORIZONS = (1.0, 1.5, 2.0)` |

`LLM_MMI_PRESET` and `LLM_MMI_TEMPERATURE` are legacy environment-variable names preserved by the current code. They control LLM-LTSP ablations and sampling exactly as shown below.

---

## 3. Setup

```bash
pip install pandas openpyxl tqdm openai numpy statsmodels
```

Optional packages for the `llm/train/` SFT/RFT utilities:

```bash
pip install lifelines transformers datasets trl peft
```

For the default manuscript-style deployment, serve InternVL3-14B through an OpenAI-compatible local endpoint, then set:

```bash
export LT_INTERNVL_BASE_URL=http://localhost:8000/v1
export LT_BACKBONE=internvl3-14b
export LT_VLM_BACKBONE=internvl3-14b
```

The inference runner expects an API-key file even for local OpenAI-compatible servers. Use one key per line; for local gateways that ignore authentication, a dummy token is sufficient.

```bash
export LT_API_KEYS=/absolute/path/to/api.txt
```

Input data:

```bash
export LT_DATA=/absolute/path/to/lt_input.xlsx
export LT_KEYFRAME_INDEX=/absolute/path/to/site1.json
export LT_OUTPUT=lt_output.xlsx
```

The Excel file must contain at least:

- `ID` — unique recipient identifier;
- `input` — harmonised English longitudinal narrative containing CT reports, biochemical results, and discharge-summary information.

For downstream Cox fusion, the outcome table must contain:

- `ID`;
- `time` — follow-up time in years, because `fusion.py` evaluates horizons at `1.0`, `1.5`, and `2.0` years;
- `event` — 1 for all-cause death and 0 for censoring.

If follow-up is stored in days, convert it to years before calling `llm.lt.fusion`. If an older table uses `mortstat`, rename it to `event`.

---

## 4. Default run: A1.0 Full LLM-LTSP

Run from the repository root:

```bash
python -m llm.lt.inference
```

This corresponds to:

- text backbone = `internvl3-14b` by default;
- VLM backbone = `internvl3-14b` by default;
- preset = `full` / A1.0;
- text branch = 15 value-emitting tasks: 10 organ/frailty biological-age dimensions, 4 LT-specific risk gradings, and 1 overall biological age;
- VLM branch = four-step CT keyframe reasoning with terminal `CT-derived risk`;
- fused representation = 16 dimensions;
- decoding = deterministic greedy sampling, `T = 0`;
- workers = `LT_WORKERS` threads, default 16, capped by API-key count and remaining row count.

Breakpoint resume is automatic. If `LT_OUTPUT` already exists, the runner resumes after the last written `ID`.

For a text-only smoke test using the bundled spreadsheet:

```bash
LLM_MMI_PRESET=A6.1 \
LT_DATA=dataset_example/example_input.xlsx \
LT_OUTPUT=example_lt_output.xlsx \
python -m llm.lt.inference
```

`A6.1` disables the CT/VLM branch. The text branch still requires a reachable OpenAI-compatible model endpoint and a valid `LT_API_KEYS` file.

---

## 5. Reproducing the ablation table

All presets are defined in `llm/lt/config.py::ABLATION_PRESETS`. Each preset changes one component relative to the full LLM-LTSP configuration.

| Ablation ID | Component touched | Reproduce with |
|---|---|---|
| **A1.0** Full LLM-LTSP | reference | `python -m llm.lt.inference` |
| **A1.1** drop Role layer | prompt | `LLM_MMI_PRESET=A1.1 python -m llm.lt.inference` |
| **A1.2** drop Knowledge layer | prompt | `LLM_MMI_PRESET=A1.2 python -m llm.lt.inference` |
| **A1.3** drop CoT protocol | prompt/reasoning | `LLM_MMI_PRESET=A1.3 python -m llm.lt.inference` |
| **A1.4** drop Output contract | prompt/free-form | `LLM_MMI_PRESET=A1.4 python -m llm.lt.inference` |
| **A2.1** decomposition `K = 1` | text DAG | `LLM_MMI_PRESET=A2.1 python -m llm.lt.inference` |
| **A2.2** decomposition `K = 11` | text DAG | `LLM_MMI_PRESET=A2.2 python -m llm.lt.inference` |
| **A2.3** flat order, no `τ0` anchor | text DAG | `LLM_MMI_PRESET=A2.3 python -m llm.lt.inference` |
| **A3.1** zero-shot, no per-task CoT | reasoning | `LLM_MMI_PRESET=A3.1 python -m llm.lt.inference` |
| **A4.1** stochastic single-shot | sampling | `LLM_MMI_PRESET=A4.1 LLM_MMI_TEMPERATURE=0.7 python -m llm.lt.inference` |
| **A4.2** majority voting `N = 5` | sampling | `LLM_MMI_PRESET=A4.2 python -m llm.lt.inference` |
| **A6.1** text-only, drop VLM branch | modality | `LLM_MMI_PRESET=A6.1 python -m llm.lt.inference` |

For `A4.2`, the `Response` field contains five text-branch JSON blocks separated by the literal string `---SAMPLE-DELIM---`. Aggregate the five samples by mode/median before fitting Cox.

For `A1.4`, the response is free-form text; post-hoc parsing is required before `fusion.py` can consume the output.

For `A6.1`, the inference output intentionally omits the CT-derived binary feature. The built-in full-fusion runner expects the 16-D full feature set, so text-only evaluation should omit `ct-derived risk` in a separate Cox call or use a text-only fusion variant.

---

## 6. Multimodal fusion and Cox survival evaluation

After inference, fit the Cox proportional-hazards model on the training centre and evaluate fixed coefficients on validation centres:

```bash
python - <<'PY'
from llm.lt.fusion import run_fusion_pipeline

metrics = run_fusion_pipeline(
    train_inference="centre1_lt_output.xlsx",
    train_outcome="centre1_outcome.xlsx",
    validation_pairs=[
        ("Centre 2", "centre2_lt_output.xlsx", "centre2_outcome.xlsx"),
        ("Centre 3", "centre3_lt_output.xlsx", "centre3_outcome.xlsx"),
    ],
    out_path="lt_fusion_metrics.xlsx",
)
print(metrics)
PY
```

`run_fusion_pipeline()` performs the following operations:

1. parses the text-branch JSON in `Response` into the 15 LLM-LTSP text variables;
2. maps LT risk grades as `low-risk = 1`, `medium-low-risk = 2`, `medium-risk = 3`, `medium-high-risk = 4`, `high-risk = 5`;
3. appends the binary `ct-derived risk` variable from the VLM branch;
4. z-scores features before Cox fitting;
5. fits Cox PH on the training cohort;
6. applies fixed coefficients and Breslow baseline survival to validation cohorts;
7. reports C-index, IPCW AUC, and IPCW Brier metrics at 12, 18, and 24 months.

The loader also accepts pre-parsed Excel/parquet/csv files where the 15 text variables already appear as named columns. The legacy column `video_risk` is accepted as an alias for `ct-derived risk` and is normalised from `{1, 5}` or `{0, 1}` to `{0, 1}`.

---

## 7. Configuration reference

| Env var | Default | Effect |
|---|---|---|
| `LLM_MMI_PRESET` | `full` | Ablation preset: `full`, `A1.1`–`A1.4`, `A2.1`–`A2.3`, `A3.1`, `A4.1`, `A4.2`, `A6.1` |
| `LLM_MMI_TEMPERATURE` | `0.7` only for `A4.1` | Sampling temperature for stochastic single-shot ablation |
| `LT_BACKBONE` | `internvl3-14b` | Text-branch frozen policy backbone |
| `LT_VLM_BACKBONE` | same as `LT_BACKBONE` | CT/VLM-branch backbone; must support vision for the full model |
| `LT_INTERNVL_BASE_URL` | `http://localhost:8000/v1` | OpenAI-compatible local InternVL endpoint |
| `LT_VLM_TOPK` | `8` | Number of CAM-ranked CT keyframes retained per acquisition phase |
| `LT_WORKERS` | `16` | Thread-pool size, capped by API-key count and row count |
| `LT_DATA` | `/home/user/Agent/LT/yanzheng11-2.xlsx` | Input Excel file with `ID` and `input` |
| `LT_KEYFRAME_INDEX` | `/home/user/Agent/LT/site1.json` | Per-recipient CT keyframe manifest |
| `LT_API_KEYS` | `/home/user/Agent/LT/api.txt` | One OpenAI-compatible API key per line |
| `LT_OUTPUT` | `lt_output.xlsx` | Main merged inference output |
| `LT_VLM_OUTPUT` | `lt_vlm_output.xlsx` | Reserved VLM output path variable |
| `LT_FUSION_OUTPUT` | `lt_fusion.xlsx` | Reserved fusion output path variable |


---

## 8. Output schema

Each row of `LT_OUTPUT` corresponds to one recipient:

| Column | Meaning |
|---|---|
| `ID` | recipient identifier |
| `input` | harmonised English longitudinal narrative |
| `Response` | raw text-branch LLM output; JSON for contract-preserving presets |
| `VLM_Response` | raw four-step CT/VLM JSON output, or null if the VLM branch is disabled or no keyframes are found |
| `ct-derived risk` | binary CT-derived risk: `1` for high-risk, `0` for low-risk |
| `ct-derived label` | parsed VLM label: `high-risk` or `low-risk` |

The standard 15 text-branch variables parsed by `llm/lt/fusion.py` are:

```text
cardiovascular system age
digestive system age
respiratory system age
endocrine/metabolic system age
nervous system age
hematologic system age
musculoskeletal/motor system age
urinary system age
immune system age
frailty age
medical history grading
rejection risk grading
infection risk grading
postoperative complication risk grading
overall biological age
```

The full LLM-LTSP Cox vector appends:

```text
ct-derived risk
```

---

## 9. CT keyframe manifest notes

The VLM branch reads the JSON file pointed to by `LT_KEYFRAME_INDEX`. For each `ID`, the manifest should provide CT phase folders under the keys used by the preprocessing code:

```json
{
  "PATIENT_ID": {
    "CT": {
      "Preoperative": {"scan_a": {"file_path": "/path/to/preop/png_folder"}},
      "Postoperative": {"scan_b": {"file_path": "/path/to/predischarge/png_folder"}},
      "Post-discharge": {"scan_c": {"file_path": "/path/to/postdischarge/png_folder"}}
    },
    "Report": {}
  }
}
```

Within each scan folder, `vlm.py` reads `.png` slices and optionally `cam_scores.json`. If `cam_scores.json` is present, slices are sorted by descending CAM score; otherwise deterministic filename order is used.

---

## 10. Citation and provenance

If you use this repository, cite the LLM-LTSP manuscript in preparation and cite the frozen policy backbone selected through `LT_BACKBONE` and `LT_VLM_BACKBONE`.
