#!/usr/bin/env python3
import argparse
import html
import json
import math
from pathlib import Path


import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from name_disambiguation.core.llm_cluster import field_value, load_truth
from name_disambiguation.core.metrics import analyze_result, b3_metrics
from name_disambiguation.paths import FIGURES_ROOT, RESULTS_ROOT, dataset_dir


DEFAULT_DATA = dataset_dir("v3")
DEFAULT_TRUTH = dataset_dir() / "sna_valid_ground_truth.json"
DEFAULT_RESULTS = RESULTS_ROOT / "baseline_clusters"
DEFAULT_OUTPUT = FIGURES_ROOT / "field_selection" / "overview"
PLOT_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "toImageButtonOptions": {"format": "png", "scale": 2},
}

METRIC_COLUMNS = [
    "name",
    "status",
    "error_reason",
    "paper_count",
    "prompt_tokens",
    "tokens_per_paper",
    "pairwise_precision",
    "pairwise_recall",
    "pairwise_f1",
    "b3_precision",
    "b3_recall",
    "b3_f1",
    "true_cluster_count",
    "predicted_cluster_count",
    "cluster_count_ratio",
    "log2_cluster_count_ratio",
    "true_singleton_ratio",
    "predicted_singleton_ratio",
    "true_largest_cluster_ratio",
    "predicted_largest_cluster_ratio",
    "true_cluster_size_entropy",
    "title_coverage",
    "organization_coverage",
    "coauthor_coverage",
    "overall_coverage",
]

HOVER_COLUMNS = [
    "paper_count",
    "pairwise_precision",
    "pairwise_recall",
    "pairwise_f1",
    "b3_precision",
    "b3_recall",
    "b3_f1",
    "true_cluster_count",
    "predicted_cluster_count",
    "cluster_count_ratio",
    "title_coverage",
    "organization_coverage",
    "coauthor_coverage",
    "overall_coverage",
]


def truth_as_dict(value):
    if isinstance(value, dict):
        return value
    if isinstance(value, list):
        return {str(index): ids for index, ids in enumerate(value)}
    raise ValueError("ground truth must be a dict or list of clusters")


def cluster_statistics(groups, paper_count):
    sizes = [len(group) for group in groups if group]
    if not sizes:
        return {
            "count": 0,
            "singleton_ratio": None,
            "largest_ratio": None,
            "entropy": None,
        }
    probabilities = [size / paper_count for size in sizes]
    return {
        "count": len(sizes),
        "singleton_ratio": sum(size == 1 for size in sizes) / len(sizes),
        "largest_ratio": max(sizes) / paper_count,
        "entropy": -sum(p * math.log2(p) for p in probabilities if p),
    }


def normalized_predicted_groups(clusters, paper_ids):
    valid = set(paper_ids)
    seen = set()
    groups = []
    for cluster in clusters:
        group = set()
        for paper_id in cluster.get("paper_ids", []):
            if paper_id in valid and paper_id not in seen:
                seen.add(paper_id)
                group.add(paper_id)
        if group:
            groups.append(group)
    groups.extend({paper_id} for paper_id in paper_ids if paper_id not in seen)
    return groups


def field_coverages(name, paper_ids, papers):
    available = [papers[paper_id] for paper_id in paper_ids if paper_id in papers]
    if not available:
        return {
            "title_coverage": None,
            "organization_coverage": None,
            "coauthor_coverage": None,
            "overall_coverage": None,
        }
    title = sum(bool(paper.get("title")) for paper in available) / len(paper_ids)
    organization = sum(
        any(field_value(paper, name, "organization"))
        for paper in available
    ) / len(paper_ids)
    coauthor = sum(
        bool(field_value(paper, name, "coauthors"))
        for paper in available
    ) / len(paper_ids)
    return {
        "title_coverage": title,
        "organization_coverage": organization,
        "coauthor_coverage": coauthor,
        "overall_coverage": (title + organization + coauthor) / 3,
    }


def analyze_block(name, paper_ids, truth, papers, result_path, combination):
    row = {column: None for column in METRIC_COLUMNS}
    row.update({
        "name": name,
        "status": "error",
        "error_reason": "",
        "paper_count": len(paper_ids),
        "combination": combination,
        "result_file": str(result_path),
    })
    if not result_path.exists():
        row.update(status="missing_result", error_reason="prediction file not found")
        return row

    try:
        result = json.loads(result_path.read_text())
        clusters = result.get("clusters")
        if not isinstance(clusters, list) or not clusters:
            raise ValueError("prediction does not contain a non-empty clusters list")
        truth_dict = truth_as_dict(truth)
        truth_groups = [set(ids) for ids in truth_dict.values()]
        predicted_groups = normalized_predicted_groups(clusters, paper_ids)
        analysis = analyze_result(clusters, paper_ids, truth_dict)
        if analysis["unknown_paper_ids"]:
            raise ValueError(
                f"{len(analysis['unknown_paper_ids'])} unknown paper IDs"
            )
        if analysis["cross_cluster_duplicates"]:
            raise ValueError(
                f"{len(analysis['cross_cluster_duplicates'])} cross-cluster duplicates"
            )

        pairwise = analysis["pairwise_metrics_missing_as_singletons"]
        b3 = b3_metrics(truth_groups, predicted_groups, paper_ids)
        true_stats = cluster_statistics(truth_groups, len(paper_ids))
        predicted_stats = cluster_statistics(predicted_groups, len(paper_ids))
        ratio = predicted_stats["count"] / true_stats["count"]
        row.update({
            "status": "completed",
            "pairwise_precision": pairwise["precision"],
            "pairwise_recall": pairwise["recall"],
            "pairwise_f1": pairwise["f1"],
            "b3_precision": b3["precision"],
            "b3_recall": b3["recall"],
            "b3_f1": b3["f1"],
            "true_cluster_count": true_stats["count"],
            "predicted_cluster_count": predicted_stats["count"],
            "cluster_count_ratio": ratio,
            "log2_cluster_count_ratio": math.log2(ratio),
            "true_singleton_ratio": true_stats["singleton_ratio"],
            "predicted_singleton_ratio": predicted_stats["singleton_ratio"],
            "true_largest_cluster_ratio": true_stats["largest_ratio"],
            "predicted_largest_cluster_ratio": predicted_stats["largest_ratio"],
            "true_cluster_size_entropy": true_stats["entropy"],
            **field_coverages(name, paper_ids, papers),
        })
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        row["error_reason"] = str(error)
    return row


def write_figure(fig, output):
    fig.update_layout(
        template="plotly_white",
        hoverlabel={"font_size": 13},
        margin={"l": 70, "r": 40, "t": 90, "b": 70},
    )
    fig.write_html(
        output,
        include_plotlyjs=True,
        full_html=True,
        config=PLOT_CONFIG,
    )


def hover_data(columns=None):
    selected = columns or HOVER_COLUMNS
    return {column: ":.4f" for column in selected}


def ranked_f1(df, metric, output):
    ranked = df.sort_values(metric).reset_index(drop=True).copy()
    ranked["rank"] = ranked.index + 1
    metric_label = metric.replace("_", " ").upper()
    line_x, line_y = [], []
    for row in ranked.itertuples():
        line_x.extend([row.rank, row.rank, None])
        line_y.extend([0, getattr(row, metric), None])
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=line_x,
        y=line_y,
        mode="lines",
        line={"color": "rgba(90,110,130,0.28)", "width": 1},
        hoverinfo="skip",
        showlegend=False,
    ))
    custom = ranked[[
        "name", "pairwise_f1", "b3_f1", "paper_count",
        "true_cluster_count", "predicted_cluster_count", "overall_coverage",
    ]].to_numpy()
    fig.add_trace(go.Scatter(
        x=ranked["rank"],
        y=ranked[metric],
        mode="markers",
        marker={
            "size": 9,
            "color": ranked[metric],
            "colorscale": "RdYlGn",
            "cmin": 0,
            "cmax": 1,
            "colorbar": {"title": metric},
        },
        customdata=custom,
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Rank: %{x}<br>"
            f"{metric}: %{{y:.4f}}<br>"
            "Pairwise F1: %{customdata[1]:.4f}<br>"
            "B³ F1: %{customdata[2]:.4f}<br>"
            "Papers: %{customdata[3]}<br>"
            "True / predicted clusters: %{customdata[4]} / %{customdata[5]}<br>"
            "Overall coverage: %{customdata[6]:.4f}<extra></extra>"
        ),
    ))
    fig.add_hline(y=0.9, line_dash="dash", line_color="firebrick")
    fig.update_layout(title=f"Author Blocks Ranked by {metric_label}")
    fig.update_xaxes(title=f"Rank (lowest to highest {metric_label})")
    fig.update_yaxes(title=metric_label, range=[0, 1.03])
    write_figure(fig, output)


def scatter_plot(
    df, x, y, title, output, color=None, size=None, hover=None,
    x_title=None, y_title=None, log_x=False, vertical_lines=None,
    horizontal_lines=None, vertical_line_labels=None,
):
    size_values = None
    if size:
        size_values = 8 + 20 * (
            df[size].fillna(0) / max(df[size].max(), 1)
            if size == "paper_count" else df[size].fillna(0)
        )
    fig = px.scatter(
        df,
        x=x,
        y=y,
        color=color,
        size=size_values,
        hover_name="name",
        hover_data=hover_data(hover),
        color_continuous_scale="RdYlGn" if color == "b3_f1" else "Viridis",
        title=title,
        labels={x: x_title or x.replace("_", " ").title(),
                y: y_title or y.replace("_", " ").title()},
        log_x=log_x,
    )
    fig.update_traces(marker={"line": {"width": 0.6, "color": "white"}})
    for value in vertical_lines or []:
        fig.add_vline(x=value, line_dash="dash", line_color="gray")
        if vertical_line_labels and value in vertical_line_labels:
            fig.add_annotation(
                x=value,
                y=1,
                yref="paper",
                text=vertical_line_labels[value],
                showarrow=False,
                xanchor="left",
                yanchor="bottom",
            )
    for value in horizontal_lines or []:
        fig.add_hline(y=value, line_dash="dash", line_color="gray")
    write_figure(fig, output)


def coverage_figure(df, metric, output):
    metric_label = metric.replace("_", " ").upper()
    columns = [
        ("overall_coverage", "Overall Coverage"),
        ("title_coverage", "Title Coverage"),
        ("organization_coverage", "Organization Coverage"),
        ("coauthor_coverage", "Coauthor Coverage"),
    ]
    usable = [(column, label) for column, label in columns if df[column].notna().any()]
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[label for _, label in usable],
    )
    for index, (column, label) in enumerate(usable):
        row, col = divmod(index, 2)
        subset = df[df[column].notna()]
        custom = subset[[
            "name", "paper_count", "true_cluster_count",
            "predicted_cluster_count", "pairwise_precision",
            "pairwise_recall", "pairwise_f1",
        ]].to_numpy()
        fig.add_trace(go.Scatter(
            x=subset[column],
            y=subset[metric],
            mode="markers",
            marker={
                "size": 9,
                "color": subset[metric],
                "colorscale": "RdYlGn",
                "cmin": 0,
                "cmax": 1,
            },
            customdata=custom,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                f"{label}: %{{x:.4f}}<br>{metric}: %{{y:.4f}}<br>"
                "Papers: %{customdata[1]}<br>"
                "True / predicted clusters: %{customdata[2]} / %{customdata[3]}<br>"
                "Pairwise P/R/F1: %{customdata[4]:.4f} / "
                "%{customdata[5]:.4f} / %{customdata[6]:.4f}<extra></extra>"
            ),
            showlegend=False,
        ), row=row + 1, col=col + 1)
        fig.update_xaxes(title_text=label, range=[-0.02, 1.02], row=row + 1, col=col + 1)
        fig.update_yaxes(title_text=metric_label, range=[-0.02, 1.02], row=row + 1, col=col + 1)
    fig.update_layout(title=f"Metadata Coverage vs {metric_label}", height=850)
    write_figure(fig, output)


def f1_comparison_figure(df, output):
    scatter_plot(
        df,
        "b3_f1",
        "pairwise_f1",
        "B³ F1 vs Pairwise F1",
        output,
        color="paper_count",
        hover=[
            "paper_count", "true_cluster_count", "predicted_cluster_count",
            "pairwise_precision", "pairwise_recall", "b3_precision", "b3_recall",
        ],
        x_title="B³ F1",
        y_title="Pairwise F1",
        vertical_lines=[0.9],
        horizontal_lines=[0.9],
    )


def comparison_figure(df, x, y, title, output, hover):
    fig = px.scatter(
        df,
        x=x,
        y=y,
        color="b3_f1",
        hover_name="name",
        hover_data=hover_data(hover),
        color_continuous_scale="RdYlGn",
        range_color=[0, 1],
        title=title,
    )
    upper = max(df[x].max(), df[y].max(), 1)
    fig.add_trace(go.Scatter(
        x=[0, upper],
        y=[0, upper],
        mode="lines",
        line={"dash": "dash", "color": "gray"},
        name="y = x",
        hoverinfo="skip",
    ))
    fig.update_xaxes(range=[0, upper * 1.03])
    fig.update_yaxes(range=[0, upper * 1.03])
    write_figure(fig, output)


def heatmap_figure(df, output):
    columns = [
        "b3_f1", "pairwise_f1", "paper_count", "true_cluster_count",
        "predicted_cluster_count", "cluster_count_ratio", "title_coverage",
        "organization_coverage", "coauthor_coverage", "overall_coverage",
        "true_singleton_ratio", "predicted_singleton_ratio",
        "true_largest_cluster_ratio", "predicted_largest_cluster_ratio",
        "true_cluster_size_entropy",
    ]
    ranked = df.sort_values("b3_f1")
    values = ranked[columns].astype(float)
    standard = values.copy()
    for column in columns:
        deviation = values[column].std(ddof=0)
        standard[column] = (
            (values[column] - values[column].mean()) / deviation
            if deviation and not math.isnan(deviation) else 0
        )
    fig = go.Figure(go.Heatmap(
        z=standard.to_numpy(),
        x=columns,
        y=ranked["name"],
        customdata=values.to_numpy(),
        colorscale="RdBu",
        zmid=0,
        hovertemplate=(
            "<b>%{y}</b><br>Metric: %{x}<br>"
            "Raw value: %{customdata:.5g}<br>"
            "Standardized: %{z:.3f}<extra></extra>"
        ),
        colorbar={"title": "z-score"},
    ))
    fig.update_layout(
        title="Standardized Block-Level Feature Heatmap (sorted by B³ F1)",
        height=max(900, 18 * len(ranked)),
    )
    write_figure(fig, output)


def message_html(output, title, message):
    output.write_text(
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{html.escape(title)}</title>"
        "<style>body{font-family:system-ui;max-width:900px;margin:70px auto;"
        "line-height:1.6;color:#1f2937} .note{padding:24px;background:#f3f4f6;"
        "border-left:5px solid #64748b}</style></head><body>"
        f"<h1>{html.escape(title)}</h1><div class='note'>{html.escape(message)}</div>"
        "</body></html>\n"
    )


def write_readme(output_dir, args):
    (output_dir / "README.md").write_text(f"""# Stage1 Interactive Analysis

Generate the analysis again with:

```bash
.venv/bin/python -m name_disambiguation.experiments.field_selection.combinations.analyze_interactive \\
  --data-dir {args.data_dir} \\
  --result-dir {args.result_dir} \\
  --ground-truth {args.ground_truth} \\
  --output-dir {args.output_dir}
```

Each point represents one author-name block. Hover over a point to inspect the
author name and block-level metrics.

Coverage definitions:

- `title_coverage`: fraction of papers with a non-empty title.
- `organization_coverage`: fraction with a non-empty organization for the target author.
- `coauthor_coverage`: fraction with at least one coauthor.
- `overall_coverage`: arithmetic mean of the three coverage values above.

Post-hoc analysis only: B3 and pairwise scores, true cluster count, true
singletons, true largest-cluster ratio, and true cluster-size entropy use ground
truth and must not be treated as deployable input features.

Potential hard-block routing features: paper count, metadata coverage, prompt
token count when logged, and predicted cluster structure.

`prompt_tokens` is empty in this run because successful API response usage was
not persisted. The token HTML explains this rather than presenting an estimate
as an observed token count.
""")


def write_index(output_dir, plot_files, df, metric, threshold):
    hardest = df[df[metric] < threshold].sort_values(metric)
    links = "\n".join(
        f"<li><a href='{html.escape(path.name)}'>{html.escape(title)}</a></li>"
        for title, path in plot_files
    )
    hard_rows = "\n".join(
        f"<li><b>{html.escape(row['name'])}</b>: {metric}={row[metric]:.4f}, "
        f"papers={int(row['paper_count'])}, true/predicted clusters="
        f"{int(row['true_cluster_count'])}/{int(row['predicted_cluster_count'])}</li>"
        for _, row in hardest.iterrows()
    ) or "<li>None</li>"
    (output_dir / "index.html").write_text(f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Stage1 Interactive Analysis</title>
<style>
body{{font-family:system-ui;max-width:1050px;margin:45px auto;line-height:1.55;color:#172033}}
a{{color:#175cd3}} code{{background:#f1f5f9;padding:2px 5px}} .note{{background:#f8fafc;padding:18px;border-left:4px solid #64748b}}
</style></head><body>
<h1>Stage1 Interactive Author-Block Analysis</h1>
<div class="note">Each point represents one author-name block. Hover over points
to view the author name and detailed metrics. Ground-truth-derived metrics are
for post-hoc analysis only.</div>
<h2>Charts</h2><ul>{links}</ul>
<h2>Hardest blocks ({html.escape(metric)} &lt; {threshold:.2f})</h2>
<ol>{hard_rows}</ol>
<p>Tabular data: <a href="analysis_summary.csv">analysis_summary.csv</a>.
Definitions and usage: <a href="README.md">README.md</a>.</p>
</body></html>
""")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_TRUTH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--combination", default="title+coauthors+organization")
    parser.add_argument("--f1-metric", default="b3_f1")
    parser.add_argument("--f1-threshold", type=float, default=0.9)
    args = parser.parse_args()

    raw = json.loads((args.data_dir / "sna_valid_raw.json").read_text())
    papers = json.loads((args.data_dir / "sna_valid_pub.json").read_text())
    truth = load_truth(args.data_dir, args.ground_truth)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        analyze_block(
            name,
            paper_ids,
            truth.get(name),
            papers,
            args.result_dir / f"{name}.json",
            args.combination,
        )
        for name, paper_ids in raw.items()
    ]
    summary = pd.DataFrame(rows)
    summary.to_csv(args.output_dir / "analysis_summary.csv", index=False)
    if args.f1_metric not in summary.columns:
        raise ValueError(f"unknown --f1-metric: {args.f1_metric}")
    complete = summary[
        (summary["status"] == "completed") & summary[args.f1_metric].notna()
    ].copy()
    if complete.empty:
        raise ValueError("no completed author blocks are available for plotting")

    plot_files = []
    def add(title, filename):
        path = args.output_dir / filename
        plot_files.append((title, path))
        return path

    ranked_f1(
        complete,
        args.f1_metric,
        add("Ranked F1 Distribution", "f1_ranked_distribution.html"),
    )
    ranked_f1(
        complete,
        "pairwise_f1",
        add("Pairwise F1 Ranked Distribution", "pairwise_f1_ranked_distribution.html"),
    )
    f1_comparison_figure(
        complete,
        add("B³ F1 vs Pairwise F1", "b3_vs_pairwise_f1.html"),
    )
    precision_recall = add(
        "Pairwise Precision vs Recall",
        "pairwise_precision_recall.html",
    )
    scatter_plot(
        complete,
        "pairwise_precision",
        "pairwise_recall",
        "Pairwise Precision vs Recall: upper-right easy; upper-left over-merging; lower-right over-splitting",
        precision_recall,
        color="b3_f1",
        size="paper_count",
        hover=HOVER_COLUMNS,
        x_title="Pairwise Precision",
        y_title="Pairwise Recall",
        vertical_lines=[0.9],
        horizontal_lines=[0.9],
    )
    coverage_figure(
        complete,
        args.f1_metric,
        add("Coverage vs F1", "coverage_vs_f1.html"),
    )
    coverage_figure(
        complete,
        "pairwise_f1",
        add("Coverage vs Pairwise F1", "pairwise_coverage_vs_f1.html"),
    )
    scatter_plot(
        complete,
        "paper_count",
        args.f1_metric,
        "Paper Count vs B³ F1",
        add("Paper Count vs F1", "paper_count_vs_f1.html"),
        color="true_cluster_count",
        size="overall_coverage",
        hover=HOVER_COLUMNS,
        x_title="Paper Count",
        y_title="B³ F1",
    )
    scatter_plot(
        complete,
        "paper_count",
        "pairwise_f1",
        "Paper Count vs Pairwise F1",
        add("Paper Count vs Pairwise F1", "pairwise_paper_count_vs_f1.html"),
        color="true_cluster_count",
        size="overall_coverage",
        hover=HOVER_COLUMNS,
        x_title="Paper Count",
        y_title="Pairwise F1",
    )
    scatter_plot(
        complete,
        "true_cluster_count",
        args.f1_metric,
        "True Cluster Count vs B³ F1",
        add("True Cluster Count vs F1", "true_cluster_count_vs_f1.html"),
        color="paper_count",
        hover=[
            "paper_count", "overall_coverage", "true_cluster_size_entropy",
            "true_largest_cluster_ratio", "pairwise_f1",
        ],
        x_title="True Cluster Count",
        y_title="B³ F1",
    )
    scatter_plot(
        complete,
        "true_cluster_count",
        "pairwise_f1",
        "True Cluster Count vs Pairwise F1",
        add("True Cluster Count vs Pairwise F1", "pairwise_true_cluster_count_vs_f1.html"),
        color="paper_count",
        hover=[
            "paper_count", "overall_coverage", "true_cluster_size_entropy",
            "true_largest_cluster_ratio", "b3_f1",
        ],
        x_title="True Cluster Count",
        y_title="Pairwise F1",
    )
    ratio_output = add("Cluster Count Ratio vs F1", "cluster_ratio_vs_f1.html")
    scatter_plot(
        complete,
        "log2_cluster_count_ratio",
        args.f1_metric,
        "Cluster Count Ratio vs B³ F1: left over-merging; right over-splitting",
        ratio_output,
        color="b3_f1",
        hover=[
            "true_cluster_count", "predicted_cluster_count",
            "true_singleton_ratio", "predicted_singleton_ratio",
            "true_largest_cluster_ratio", "predicted_largest_cluster_ratio",
            "pairwise_precision", "pairwise_recall",
        ],
        x_title="log2(Predicted Cluster Count / True Cluster Count)",
        y_title="B³ F1",
        vertical_lines=[0],
        vertical_line_labels={0: "predicted = true"},
    )
    scatter_plot(
        complete,
        "log2_cluster_count_ratio",
        "pairwise_f1",
        "Cluster Count Ratio vs Pairwise F1: left over-merging; right over-splitting",
        add("Cluster Count Ratio vs Pairwise F1", "pairwise_cluster_ratio_vs_f1.html"),
        color="pairwise_f1",
        hover=[
            "true_cluster_count", "predicted_cluster_count",
            "true_singleton_ratio", "predicted_singleton_ratio",
            "true_largest_cluster_ratio", "predicted_largest_cluster_ratio",
            "b3_f1", "pairwise_precision", "pairwise_recall",
        ],
        x_title="log2(Predicted Cluster Count / True Cluster Count)",
        y_title="Pairwise F1",
        vertical_lines=[0],
        vertical_line_labels={0: "predicted = true"},
    )
    token_output = add("Token Count vs F1", "token_count_vs_f1.html")
    if complete["prompt_tokens"].notna().any():
        scatter_plot(
            complete[complete["prompt_tokens"].notna()],
            "prompt_tokens",
            args.f1_metric,
            "Prompt Tokens vs B³ F1",
            token_output,
            color="true_cluster_count",
            hover=["paper_count", "overall_coverage", "true_cluster_count"],
        )
    else:
        message_html(
            token_output,
            "Token Count vs B³ F1",
            "Reliable prompt token usage was not persisted for successful API calls, "
            "so no token plot was generated. prompt_tokens and tokens_per_paper remain "
            "empty in analysis_summary.csv.",
        )
    comparison_figure(
        complete,
        "true_singleton_ratio",
        "predicted_singleton_ratio",
        "True vs Predicted Singleton-Cluster Ratio",
        add("Singleton Ratio Comparison", "singleton_ratio_comparison.html"),
        ["b3_f1", "cluster_count_ratio", "paper_count"],
    )
    comparison_figure(
        complete,
        "true_largest_cluster_ratio",
        "predicted_largest_cluster_ratio",
        "True vs Predicted Largest-Cluster Ratio",
        add("Largest Cluster Ratio Comparison", "largest_cluster_ratio_comparison.html"),
        ["b3_f1", "pairwise_precision", "pairwise_recall", "paper_count"],
    )
    entropy = complete[complete["true_cluster_size_entropy"].notna()]
    entropy_output = add("True Entropy vs F1", "true_entropy_vs_f1.html")
    if entropy.empty:
        message_html(
            entropy_output,
            "True Cluster-Size Entropy vs B³ F1",
            "Ground-truth cluster-size entropy could not be computed.",
        )
    else:
        scatter_plot(
            entropy,
            "true_cluster_size_entropy",
            args.f1_metric,
            "True Cluster-Size Entropy vs B³ F1",
            entropy_output,
            color="true_cluster_count",
            size="paper_count",
            hover=[
                "true_cluster_count", "paper_count",
                "true_largest_cluster_ratio", "overall_coverage",
            ],
            x_title="True Cluster-Size Entropy (bits)",
            y_title="B³ F1",
        )
        scatter_plot(
            entropy,
            "true_cluster_size_entropy",
            "pairwise_f1",
            "True Cluster-Size Entropy vs Pairwise F1",
            add("True Entropy vs Pairwise F1", "pairwise_true_entropy_vs_f1.html"),
            color="true_cluster_count",
            size="paper_count",
            hover=[
                "true_cluster_count", "paper_count",
                "true_largest_cluster_ratio", "overall_coverage", "b3_f1",
            ],
            x_title="True Cluster-Size Entropy (bits)",
            y_title="Pairwise F1",
        )
    heatmap_figure(
        complete,
        add("Feature Heatmap", "feature_heatmap.html"),
    )
    write_readme(args.output_dir, args)
    write_index(
        args.output_dir,
        plot_files,
        complete,
        args.f1_metric,
        args.f1_threshold,
    )
    failures = summary[summary["status"] != "completed"]
    print(f"summary rows: {len(summary)}")
    print(f"plotted blocks: {len(complete)}")
    print(f"skipped blocks: {len(failures)}")
    print(f"saved: {args.output_dir}")


if __name__ == "__main__":
    main()
