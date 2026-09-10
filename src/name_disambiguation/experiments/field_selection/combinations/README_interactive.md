# Interactive Stage1 Analysis

Run:

```bash
.venv/bin/python -m name_disambiguation.experiments.field_selection.combinations.analyze_interactive \
  --data-dir data/whoiswho/data/v3/SND/valid \
  --result-dir artifacts/results/baseline_clusters \
  --ground-truth data/whoiswho/data/NA_Demo/SND/valid/sna_valid_ground_truth.json \
  --output-dir artifacts/figures/field_selection/overview
```

The command writes `analysis_summary.csv`, standalone Plotly HTML charts,
`index.html`, and a README with metric and coverage definitions.

Ground-truth-derived metrics such as F1, true cluster count, and true entropy
are post-hoc analysis features. They must not be used as deployable routing
inputs. Paper count, metadata coverage, token usage when logged, and predicted
cluster structure are potential future routing features.
