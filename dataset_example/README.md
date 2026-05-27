# Example data

`example_input.xlsx` is a small example spreadsheet for smoke testing the Python LLM-LTSP inference code.

Required columns for inference:

- `ID`: participant identifier.
- `input`: the standardized natural-language health report supplied to the LLM.

Additional columns such as `mortstat`, `time`, `Age`, and `Sex` are included only to illustrate the downstream validation schema.

The full cohort-level are not included in this code upload.
