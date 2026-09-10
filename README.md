# LLM-Based From-Scratch Author Name Disambiguation

A research repository for clustering papers written under the same ambiguous
author name. The project studies not only whether an LLM can perform scholarly
name disambiguation, but how its input fields, context structure, prompt design,
and reliability controls affect clustering quality and cost.

## Project At A Glance

| Item | Summary |
|:---:|:---:|
| Problem | Split each same-name paper block into clusters representing real authors |
| Benchmark | WhoIsWho SND, using Pairwise F1 and B³ F1 |
| Main LLM input | title + coauthors + organization (T+O+C) |
| Best pilot result | 0.9792 macro Pairwise F1, 0.9814 macro B³ F1 |
| 80-block validation | 0.7852 mean Pairwise F1, 0.8292 mean B³ F1 |
| Main failure mode | Over-splitting large or structurally difficult blocks into singletons |
| Research extension | Pre-LLM reliability routing and post-LLM result validation |

## What I Worked On

I led the LLM-oriented investigation and organized it around four connected
questions:

1. **Which evidence should the model see?** I evaluated title, abstract,
   keywords, coauthors, organization, and venue individually and in multi-field
   combinations. T+O+C offered the strongest quality-to-token trade-off in the
   pilot study.
2. **How should the same evidence be organized?** I compared flat, shuffled,
   chronological, organization-grouped, coauthor-network, and multi-field
   similarity contexts while controlling the paper content.
3. **How should the task be prompted?** I compared minimal instructions, task
   definitions, evidence guidance, error-aware constraints, and
   analyze-then-cluster prompting.
4. **When should an LLM result be trusted?** I built an exploratory reliability
   router from 27 pre-call block features and analyzed post-call symptoms such
   as singleton inflation, missing IDs, and implausible cluster structure.

Before these experiments, I studied three representative graph and clustering
approaches: **WhoIsWho**, **BOND**, and **MCCG**. This established the traditional
pipeline of engineered similarities, graph representation learning, and density
clustering against which the LLM approach was compared.

## Findings

- **Complementary evidence matters.** Titles provide semantic evidence, while
  coauthors and organizations provide relational evidence. Their combination
  was more reliable than relying on one evidence family alone.
- **More context is not automatically better.** Long abstracts increase token
  use and output risk. Structured reorderings sometimes helped individual
  blocks, but did not consistently dominate a deterministic flat context.
- **Output engineering is part of the method.** Compact index-to-cluster
  assignments, temperature 0, completeness checks, and duplicate detection
  substantially reduce malformed or incomplete results.
- **Failures are concentrated rather than uniform.** Median scores on 80 v3
  blocks were higher than means, indicating a strong majority alongside a small
  set of severe over-splitting failures.
- **Some failures are recoverable.** Under the recommended invocation,
  `zheng_hu` improved from 0.8663 to 0.9874 Pairwise F1 and `yi_qian` from 0 to
  0.9647. These are successful case studies, not a claim of universal recovery.

Curated metrics are available in
[`artifacts/results/portfolio/key_results.json`](artifacts/results/portfolio/key_results.json).
The full Chinese research report is
[`docs/reports/final/llm_name_disambiguation_overall_report_zh.md`](docs/reports/final/llm_name_disambiguation_overall_report_zh.md).

## Technical Design

The implementation separates reusable clustering and evaluation logic from
experiment orchestration:

```text
src/name_disambiguation/
├── core/                    # Prompt construction, response normalization, metrics
├── evaluation/              # WhoIsWho-compatible Pairwise evaluation
├── experiments/
│   ├── field_selection/     # Single-field and combination studies
│   ├── context_organization/# Input ordering and grouping studies
│   ├── prompt_design/       # Controlled prompt ablations
│   ├── stability/           # Repeated-run stability evaluation
│   └── intra_cluster/       # Experimental local refinement
├── reliability/             # Pre-LLM reliability router
├── data_io.py
└── paths.py                 # Central repository paths
```

The code uses structured JSON outputs, validates every paper ID exactly once,
treats missing predictions as singleton clusters during evaluation, and keeps
API configuration in environment variables. Experiment scripts share the same
prompt, normalization, and metric implementations to avoid silent evaluation
drift.

## Repository Layout

| Path | Purpose |
|:---:|:---:|
| `src/name_disambiguation/` | Maintained Python package |
| `tests/` | API-free regression tests for metrics and output handling |
| `docs/reports/` | Research reports and experiment interpretation |
| `docs/prompts/` | Original experiment specifications |
| `artifacts/results/` | Local experiment outputs; only curated summaries are versioned |
| `artifacts/figures/` | Local interactive Plotly analyses |
| `data/` | Local WhoIsWho data, excluded from version control |
| `references/` | Citation index and local third-party research material |

See [`docs/REPOSITORY_MAP.md`](docs/REPOSITORY_MAP.md) for a detailed map.

## Setup

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Place the WhoIsWho files under the structure documented in
[`data/README.md`](data/README.md). For an LLM call, configure one supported
provider locally:

```bash
export DEEPSEEK_API_KEY="..."
export DEEPSEEK_BASE_URL="..."
```

No key or endpoint is stored in the repository.

## Reproduce A Clustering Run

Inspect a request without spending API tokens:

```bash
python -m name_disambiguation.core.llm_cluster \
  --data-dir data/whoiswho/data/NA_Demo/SND/valid \
  --name haifeng_qian \
  --field title --field coauthors --field organization \
  --provider deepseek --dry-run
```

Run the request by removing `--dry-run`, then evaluate a benchmark-format
prediction:

```bash
python -m name_disambiguation.evaluation.snd_eval \
  --predict PATH_TO_PREDICTION.json \
  --ground-truth data/whoiswho/data/NA_Demo/SND/valid/sna_valid_ground_truth.json \
  --per-name
```

Run local checks with:

```bash
python -m unittest discover -s tests -v
```

## Scope And Limitations

The conclusions are empirical and model-dependent. Large blocks, sparse
metadata, domain shifts, and API output limits can still cause severe
over-splitting. The reliability router found useful pre-call signals, especially
block size and coauthor structure, but did not yet generalize at non-zero
coverage on the frozen 16-block test split. It is therefore presented as a
research direction rather than a production classifier.

## License And Attribution

Original code in this repository is released under the [MIT License](LICENSE).
WhoIsWho data and third-party implementations retain their original licenses
and are not redistributed by the public repository. See
[`references/README.md`](references/README.md) for the papers studied.
