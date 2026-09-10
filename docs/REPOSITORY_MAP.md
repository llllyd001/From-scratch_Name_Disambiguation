# Repository Map

## Maintained Code

- `src/name_disambiguation/core/`: shared prompt construction, response
  normalization, validation, Pairwise metrics, and B³ metrics
- `src/name_disambiguation/evaluation/`: benchmark-compatible SND evaluator
- `src/name_disambiguation/experiments/field_selection/`: single-field,
  multi-field, and field-contribution studies
- `src/name_disambiguation/experiments/context_organization/`: controlled
  context ordering and grouping experiment
- `src/name_disambiguation/experiments/prompt_design/`: controlled prompt
  ablation experiment
- `src/name_disambiguation/experiments/stability/`: repeated LLM calls and
  run-to-run comparison
- `src/name_disambiguation/experiments/intra_cluster/`: exploratory refinement
  of broad clusters
- `src/name_disambiguation/reliability/`: pre-call reliability feature
  extraction, model training, and risk-coverage analysis

## Research Documents

- `docs/reports/final/`: overall Chinese technical report and meeting outline
- `docs/reports/field_selection/`: field-combination pilot report
- `docs/reports/context_organization/`: information-organization report
- `docs/reports/prompt_design/`: prompt design report
- `docs/reports/router/`: reliability routing reports
- `docs/prompts/`: experiment requirements preserved as research provenance

## Local-Only Material

The full WhoIsWho dataset, third-party repositories, PDFs, raw API responses,
interactive figures, and large experiment outputs remain available in the local
workspace but are excluded from the public repository. This keeps the portfolio
focused on authored code, methodology, reproducible entry points, and curated
evidence without redistributing external data or generated bulk files.
