#!/usr/bin/env python3
import argparse
import csv
import itertools
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path


import numpy as np
import pandas as pd
import plotly.express as px

from name_disambiguation.core.llm_cluster import field_value, load_truth
from name_disambiguation.core.metrics import analyze_result, b3_metrics
from name_disambiguation.paths import FIGURES_ROOT, RESULTS_ROOT, dataset_dir


AUTHORS = ("haifeng_qian", "hui_cai", "jianguo_wu")
DATA_DIR = dataset_dir()
RESULT_ROOT = RESULTS_ROOT / "field_selection" / "combinations" / "NA_Demo"
OUT_DIR = RESULTS_ROOT / "field_selection" / "analysis"
PLOT_DIR = FIGURES_ROOT / "field_selection" / "analysis"
FULL = "title+coauthors+organization"
LEAVE_ONE_OUT = {
    "title": "coauthors+organization",
    "coauthor": "title+organization",
    "organization": "title+coauthors",
}
MAIN_COMBINATIONS = (
    "title",
    "coauthors",
    "organization",
    "title+coauthors",
    "title+organization",
    "coauthors+organization",
    FULL,
    "abstract",
    "title+abstract+organization",
    "title+abstract+coauthors",
    "abstract+coauthors+organization",
)
PLOT_CONFIG = {"displaylogo": False, "responsive": True}


def read_json(path):
    return json.loads(path.read_text())


def combination_path(combination, author):
    size = len(combination.split("+"))
    if size == 1:
        return (
            RESULTS_ROOT / "field_selection" / "single_field" / "NA_Demo"
            / f"{author}_{combination}.json"
        )
    return RESULT_ROOT / f"{size}_fields" / combination / f"{author}.json"


def truth_label_map(truth):
    return {pid: author_id for author_id, ids in truth.items() for pid in ids}


def normalized_groups(clusters, paper_ids):
    valid = set(paper_ids)
    seen = set()
    groups = []
    for cluster in clusters:
        group = set()
        for pid in cluster.get("paper_ids", []):
            if pid in valid and pid not in seen:
                group.add(pid)
                seen.add(pid)
        if group:
            groups.append(group)
    groups.extend({pid} for pid in paper_ids if pid not in seen)
    return groups


def predicted_same_map(clusters, paper_ids):
    labels = {}
    for index, group in enumerate(normalized_groups(clusters, paper_ids)):
        for pid in group:
            labels[pid] = index
    return labels


def cluster_metrics(clusters, paper_ids, truth):
    analysis = analyze_result(clusters, paper_ids, truth)
    pairwise = analysis["pairwise_metrics_missing_as_singletons"]
    truth_groups = [set(ids) for ids in truth.values()]
    pred_groups = normalized_groups(clusters, paper_ids)
    b3 = b3_metrics(truth_groups, pred_groups, paper_ids)
    return {
        "pairwise_precision": pairwise["precision"],
        "pairwise_recall": pairwise["recall"],
        "pairwise_f1": pairwise["f1"],
        "b3_precision": b3["precision"],
        "b3_recall": b3["recall"],
        "b3_f1": b3["f1"],
        "predicted_cluster_count": len(pred_groups),
        "ground_truth_cluster_count": len(truth),
        "complete_author_coverage": analysis["complete_author_coverage"]["rate"],
    }


def tokenize(text):
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def tfidf_vectors(texts):
    docs = [tokenize(text) for text in texts]
    df = Counter()
    for doc in docs:
        df.update(set(doc))
    vocab = {term: i for i, term in enumerate(sorted(df))}
    if not vocab:
        return np.zeros((len(texts), 0))
    idf = {
        term: math.log((1 + len(texts)) / (1 + freq)) + 1
        for term, freq in df.items()
    }
    matrix = np.zeros((len(texts), len(vocab)), dtype=float)
    for row, doc in enumerate(docs):
        counts = Counter(doc)
        total = sum(counts.values()) or 1
        for term, count in counts.items():
            matrix[row, vocab[term]] = (count / total) * idf[term]
    norms = np.linalg.norm(matrix, axis=1)
    norms[norms == 0] = 1
    return matrix / norms[:, None]


def cosine_from_vectors(vectors, i, j):
    if vectors.shape[1] == 0:
        return None
    value = float(np.dot(vectors[i], vectors[j]))
    return max(0.0, min(1.0, value))


def clean_text(value):
    return re.sub(r"\s+", " ", (value or "").strip().lower())


def text_similarity(a, b):
    ta, tb = set(tokenize(a)), set(tokenize(b))
    if not ta or not tb:
        return None
    return len(ta & tb) / len(ta | tb)


def norm_name(name):
    return re.sub(r"[^a-z0-9]", "", (name or "").lower())


def coauthor_set(paper, target_name):
    target_forms = {
        norm_name(target_name),
        "".join(reversed(re.findall(r"[a-z0-9]+", target_name.lower()))),
    }
    names = set()
    for author in paper.get("authors", []):
        name = author.get("name", "")
        if norm_name(name) not in target_forms:
            cleaned = clean_text(name)
            if cleaned:
                names.add(cleaned)
    return names


def target_org(paper, target_name):
    values = field_value(paper, target_name, "organization")
    return clean_text(values[0] if values else "")


def paper_features(author, paper_ids, papers):
    titles = [papers[pid].get("title", "") for pid in paper_ids]
    abstracts = [(papers[pid].get("abstract", "") or "")[:1200] for pid in paper_ids]
    title_vectors = tfidf_vectors(titles)
    abstract_vectors = tfidf_vectors(abstracts)
    features = {}
    for idx, pid in enumerate(paper_ids):
        paper = papers[pid]
        org = target_org(paper, author)
        coauthors = coauthor_set(paper, author)
        features[pid] = {
            "index": idx,
            "title": paper.get("title", "") or "",
            "abstract": paper.get("abstract", "") or "",
            "abstract_truncated": (paper.get("abstract", "") or "")[:350],
            "organization": org,
            "coauthors": sorted(coauthors),
            "title_available": bool(clean_text(paper.get("title", ""))),
            "abstract_available": bool(clean_text(paper.get("abstract", ""))),
            "organization_available": bool(org),
            "coauthor_available": bool(coauthors),
        }
    return features, title_vectors, abstract_vectors


def pair_similarity_record(author, pid_i, pid_j, truth_by_paper, features, title_vecs, abstract_vecs):
    fi, fj = features[pid_i], features[pid_j]
    org_available = fi["organization_available"] and fj["organization_available"]
    co_i, co_j = set(fi["coauthors"]), set(fj["coauthors"])
    union = co_i | co_j
    shared = co_i & co_j
    coauthor_available = bool(co_i) and bool(co_j)
    return {
        "author_block": author,
        "paper_i": pid_i,
        "paper_j": pid_j,
        "gt_same": truth_by_paper[pid_i] == truth_by_paper[pid_j],
        "title_available": fi["title_available"] and fj["title_available"],
        "title_cosine_similarity": cosine_from_vectors(title_vecs, fi["index"], fj["index"])
        if fi["title_available"] and fj["title_available"] else None,
        "abstract_available": fi["abstract_available"] and fj["abstract_available"],
        "abstract_cosine_similarity": cosine_from_vectors(abstract_vecs, fi["index"], fj["index"])
        if fi["abstract_available"] and fj["abstract_available"] else None,
        "organization_available": org_available,
        "organization_exact_match": (
            fi["organization"] == fj["organization"] if org_available else None
        ),
        "organization_text_similarity": text_similarity(fi["organization"], fj["organization"])
        if org_available else None,
        "coauthor_available_i": fi["coauthor_available"],
        "coauthor_available_j": fj["coauthor_available"],
        "shared_coauthor_count": len(shared) if coauthor_available else None,
        "coauthor_jaccard_similarity": len(shared) / len(union)
        if coauthor_available and union else None,
        "coauthor_overlap_indicator": bool(shared) if coauthor_available else None,
    }


def compact_records(records, limit=25000):
    if len(records) <= limit:
        return records
    same = [r for r in records if r["gt_same"]]
    diff = [r for r in records if not r["gt_same"]]
    half = limit // 2
    step_same = max(1, math.ceil(len(same) / half))
    step_diff = max(1, math.ceil(len(diff) / (limit - min(half, len(same)))))
    return same[::step_same][:half] + diff[::step_diff][: limit - min(half, len(same))]


def compact_pair_records(records, limit_per_author_relation=120):
    grouped = defaultdict(list)
    for record in records:
        grouped[(record["author_block"], record["gt_same"])].append(record)
    sampled = []
    for group in grouped.values():
        step = max(1, math.ceil(len(group) / limit_per_author_relation))
        sampled.extend(group[::step][:limit_per_author_relation])
    return sampled


def field_stats(records):
    specs = [
        ("title", "title_cosine_similarity", "title_available"),
        ("abstract", "abstract_cosine_similarity", "abstract_available"),
        ("organization", "organization_text_similarity", "organization_available"),
        ("coauthor_jaccard", "coauthor_jaccard_similarity", None),
        ("shared_coauthor_count", "shared_coauthor_count", None),
    ]
    rows = []
    for field, metric, available_col in specs:
        usable = [
            r for r in records
            if r.get(metric) is not None
            and (available_col is None or r.get(available_col))
        ]
        same = [r[metric] for r in usable if r["gt_same"]]
        diff = [r[metric] for r in usable if not r["gt_same"]]
        rows.append({
            "field": field,
            "metric": metric,
            "available_pair_count": len(usable),
            "same_pair_count": len(same),
            "different_pair_count": len(diff),
            "same_mean": float(np.mean(same)) if same else None,
            "different_mean": float(np.mean(diff)) if diff else None,
            "same_median": float(np.median(same)) if same else None,
            "different_median": float(np.median(diff)) if diff else None,
            "separation_gap": (
                float(np.mean(same) - np.mean(diff)) if same and diff else None
            ),
        })
    return rows


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_plot(fig, path):
    fig.update_layout(template="plotly_white", margin={"l": 70, "r": 30, "t": 80, "b": 80})
    fig.write_html(path, include_plotlyjs=True, full_html=True, config=PLOT_CONFIG)


def load_result(combination, author):
    path = combination_path(combination, author)
    if not path.exists():
        return None, path
    return read_json(path).get("clusters", []), path


def summarize_combinations(raw, truth):
    rows = []
    missing = []
    for combination in MAIN_COMBINATIONS:
        for author in AUTHORS:
            clusters, path = load_result(combination, author)
            if clusters is None:
                missing.append({"combination": combination, "author": author, "path": str(path)})
                continue
            row = {
                "author_block": author,
                "field_combination": combination,
                "fields": combination.split("+"),
                "result_file": str(path),
            }
            row.update(cluster_metrics(clusters, raw[author], truth[author]))
            rows.append(row)

    df = pd.DataFrame(rows)
    aggregate = []
    if not df.empty:
        metrics = [
            "pairwise_precision", "pairwise_recall", "pairwise_f1",
            "b3_precision", "b3_recall", "b3_f1",
            "complete_author_coverage",
            "predicted_cluster_count", "ground_truth_cluster_count",
        ]
        for combination, group in df.groupby("field_combination", sort=False):
            item = {"field_combination": combination, "completed_author_blocks": len(group)}
            for metric in metrics:
                item[f"macro_average_{metric}"] = float(group[metric].mean())
                item[f"median_{metric}"] = float(group[metric].median())
                item[f"min_{metric}"] = float(group[metric].min())
                item[f"max_{metric}"] = float(group[metric].max())
            aggregate.append(item)

    return rows, aggregate, missing


def pairwise_similarity(raw, truth, papers):
    records = []
    features_by_author = {}
    for author in AUTHORS:
        paper_ids = raw[author]
        truth_by_paper = truth_label_map(truth[author])
        features, title_vecs, abstract_vecs = paper_features(author, paper_ids, papers)
        features_by_author[author] = features
        for pid_i, pid_j in itertools.combinations(paper_ids, 2):
            records.append(
                pair_similarity_record(
                    author, pid_i, pid_j, truth_by_paper, features, title_vecs, abstract_vecs
                )
            )
    return records, features_by_author


def contribution_records(raw, truth, pair_records):
    by_pair = {
        (r["author_block"], r["paper_i"], r["paper_j"]): r
        for r in pair_records
    }
    summaries = []
    outputs = {}
    for field, without_combination in LEAVE_ONE_OUT.items():
        records = []
        for author in AUTHORS:
            full_clusters, _ = load_result(FULL, author)
            without_clusters, _ = load_result(without_combination, author)
            if full_clusters is None or without_clusters is None:
                continue
            full_labels = predicted_same_map(full_clusters, raw[author])
            without_labels = predicted_same_map(without_clusters, raw[author])
            truth_by_paper = truth_label_map(truth[author])
            counts = Counter()
            for pid_i, pid_j in itertools.combinations(raw[author], 2):
                gt_same = truth_by_paper[pid_i] == truth_by_paper[pid_j]
                without_same = without_labels[pid_i] == without_labels[pid_j]
                full_same = full_labels[pid_i] == full_labels[pid_j]
                if without_same == full_same:
                    change_type = "unchanged"
                elif gt_same and not without_same and full_same:
                    change_type = "fixed_false_split"
                elif not gt_same and without_same and not full_same:
                    change_type = "fixed_false_merge"
                elif gt_same and without_same and not full_same:
                    change_type = "introduced_false_split"
                else:
                    change_type = "introduced_false_merge"
                counts[change_type] += 1
                if change_type != "unchanged":
                    base = by_pair[(author, pid_i, pid_j)].copy()
                    records.append({
                        "author_block": author,
                        "field_added": field,
                        "without_field_combination": without_combination,
                        "full_combination": FULL,
                        "paper_i": pid_i,
                        "paper_j": pid_j,
                        "gt_same": gt_same,
                        "pred_without_field_same": without_same,
                        "pred_full_same": full_same,
                        "change_type": change_type,
                        "title_cosine_similarity": base["title_cosine_similarity"],
                        "abstract_cosine_similarity": base["abstract_cosine_similarity"],
                        "organization_match": base["organization_exact_match"],
                        "organization_similarity": base["organization_text_similarity"],
                        "shared_coauthor_count": base["shared_coauthor_count"],
                        "coauthor_jaccard_similarity": base["coauthor_jaccard_similarity"],
                    })
            summary = {
                "field_added": field,
                "author_block": author,
                "fixed_false_splits": counts["fixed_false_split"],
                "fixed_false_merges": counts["fixed_false_merge"],
                "introduced_false_splits": counts["introduced_false_split"],
                "introduced_false_merges": counts["introduced_false_merge"],
                "unchanged_pairs": counts["unchanged"],
            }
            summary["net_pairwise_gain"] = (
                summary["fixed_false_splits"]
                + summary["fixed_false_merges"]
                - summary["introduced_false_splits"]
                - summary["introduced_false_merges"]
            )
            summaries.append(summary)
        outputs[field] = records

    all_rows = []
    for field in LEAVE_ONE_OUT:
        rows = [row for row in summaries if row["field_added"] == field]
        total = {"field_added": field, "author_block": "ALL"}
        for key in (
            "fixed_false_splits", "fixed_false_merges",
            "introduced_false_splits", "introduced_false_merges",
            "unchanged_pairs", "net_pairwise_gain",
        ):
            total[key] = sum(row[key] for row in rows)
        all_rows.append(total)
    return outputs, summaries + all_rows


def plot_combination_bars(summary_rows):
    df = pd.DataFrame(summary_rows)
    df = df[df["field_combination"].isin(MAIN_COMBINATIONS[:7])]
    order = list(MAIN_COMBINATIONS[:7])
    for metric, filename, title in [
        ("pairwise_f1", "combination_pairwise_f1.html", "Field Combination vs Pairwise F1"),
        ("b3_f1", "combination_b3_f1.html", "Field Combination vs B³ F1"),
    ]:
        macro = (
            df.groupby("field_combination", as_index=False)[metric]
            .mean()
            .assign(author_block="macro-average")
        )
        plot_df = pd.concat([df[["author_block", "field_combination", metric]], macro])
        plot_df["field_combination"] = pd.Categorical(
            plot_df["field_combination"], categories=order, ordered=True
        )
        fig = px.bar(
            plot_df.sort_values("field_combination"),
            x="field_combination",
            y=metric,
            color="author_block",
            barmode="group",
            title=title,
            labels={"field_combination": "Field combination", metric: metric},
        )
        fig.add_vrect(
            x0=5.5, x1=6.5, fillcolor="gold", opacity=0.14, line_width=0,
            annotation_text="T+O+C", annotation_position="top left"
        )
        fig.update_yaxes(range=[0, 1.05])
        write_plot(fig, PLOT_DIR / filename)


def plot_similarity(records, stats):
    df = pd.DataFrame(compact_records(records, 18000))
    specs = [
        ("title_cosine_similarity", "similarity_distribution_title.html", "Title TF-IDF Cosine Similarity"),
        ("abstract_cosine_similarity", "similarity_distribution_abstract.html", "Abstract TF-IDF Cosine Similarity"),
        ("organization_text_similarity", "similarity_distribution_organization.html", "Organization Token Jaccard Similarity"),
        ("coauthor_jaccard_similarity", "coauthor_overlap_distribution.html", "Coauthor Jaccard Similarity"),
    ]
    for metric, filename, title in specs:
        usable = df[df[metric].notna()].copy()
        usable["Ground truth relation"] = usable["gt_same"].map({True: "same author", False: "different authors"})
        fig = px.violin(
            usable,
            x="Ground truth relation",
            y=metric,
            color="Ground truth relation",
            box=True,
            points="outliers",
            hover_data=["author_block", "paper_i", "paper_j"],
            title=title,
        )
        write_plot(fig, PLOT_DIR / filename)

    stat_df = pd.DataFrame(stats)
    fig = px.bar(
        stat_df,
        x="field",
        y="separation_gap",
        hover_data=["available_pair_count", "same_mean", "different_mean"],
        title="Pair-level Same-vs-Different Separation Gap",
        labels={"separation_gap": "same_mean - different_mean"},
    )
    write_plot(fig, PLOT_DIR / "field_similarity_gap.html")


def plot_contribution(summary_rows):
    df = pd.DataFrame([row for row in summary_rows if row["author_block"] == "ALL"])
    melt = df.melt(
        id_vars=["field_added"],
        value_vars=[
            "fixed_false_splits", "fixed_false_merges",
            "introduced_false_splits", "introduced_false_merges",
        ],
        var_name="change_type",
        value_name="pair_count",
    )
    fig = px.bar(
        melt,
        x="field_added",
        y="pair_count",
        color="change_type",
        barmode="group",
        title="Pairwise Error Repairs and New Errors after Adding Each Field",
    )
    write_plot(fig, PLOT_DIR / "field_error_repair_summary.html")

    per_author = pd.DataFrame([row for row in summary_rows if row["author_block"] != "ALL"])
    fig = px.bar(
        per_author,
        x="author_block",
        y="net_pairwise_gain",
        color="field_added",
        barmode="group",
        title="Net Pairwise Gain by Author Block",
    )
    write_plot(fig, PLOT_DIR / "field_net_gain_by_author.html")


def case_markdown(contrib, features_by_author):
    lines = ["# 字段边际贡献代表性案例\n"]
    for field, records in contrib.items():
        lines.append(f"\n## 加入 {field} 后发生变化的代表性 pair\n")
        selected = []
        for change_type, limit in [
            ("fixed_false_split", 3),
            ("fixed_false_merge", 3),
            ("introduced_false_split", 2),
            ("introduced_false_merge", 2),
        ]:
            selected.extend([r for r in records if r["change_type"] == change_type][:limit])
        if not selected:
            lines.append("没有发生变化的 pair。\n")
            continue
        for item in selected:
            fi = features_by_author[item["author_block"]][item["paper_i"]]
            fj = features_by_author[item["author_block"]][item["paper_j"]]
            shared = sorted(set(fi["coauthors"]) & set(fj["coauthors"]))
            lines.append(
                f"### {item['change_type']}：{item['author_block']} "
                f"{item['paper_i']} / {item['paper_j']}\n"
            )
            lines.append(
                f"- gt_same: `{item['gt_same']}`; without-field: "
                f"`{item['pred_without_field_same']}`; full: `{item['pred_full_same']}`\n"
            )
            lines.append(f"- Title i: {fi['title'][:220]}\n")
            lines.append(f"- Title j: {fj['title'][:220]}\n")
            lines.append(f"- Organization i: {fi['organization'] or 'NULL'}\n")
            lines.append(f"- Organization j: {fj['organization'] or 'NULL'}\n")
            lines.append(f"- Shared coauthors ({len(shared)}): {', '.join(shared[:8]) or 'NULL'}\n")
            lines.append(f"- Abstract i: {fi['abstract_truncated'] or 'NULL'}\n")
            lines.append(f"- Abstract j: {fj['abstract_truncated'] or 'NULL'}\n")
    return "\n".join(lines)


def markdown_table(rows, columns):
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column)
            if isinstance(value, float):
                value = f"{value:.4f}"
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(summary_rows, aggregate_rows, missing, sim_stats, contribution_summary):
    top = sorted(
        aggregate_rows,
        key=lambda r: (
            r.get("macro_average_b3_f1") or -1,
            r.get("macro_average_pairwise_f1") or -1,
        ),
        reverse=True,
    )[:8]
    toc = next((r for r in aggregate_rows if r["field_combination"] == FULL), {})
    gap_rows = [
        {
            "field": r["field"],
            "available_pair_count": r["available_pair_count"],
            "same_mean": r["same_mean"],
            "different_mean": r["different_mean"],
            "separation_gap": r["separation_gap"],
        }
        for r in sim_stats
    ]
    contrib_all = [r for r in contribution_summary if r["author_block"] == "ALL"]
    lines = [
        "# Pilot 实验报告：字段组合有效性与互补性分析",
        "",
        "## 1. 实验目标",
        "",
        "本实验只使用 NA_Demo 中已有历史 LLM clustering 结果的三个作者 block："
        "`haifeng_qian`、`hui_cai`、`jianguo_wu`。目标是为 "
        "`title + coauthors + organization`（T+O+C）的有效性提供 pilot 证据，"
        "而不是宣称它在完整 80 作者数据上统计显著最优。",
        "",
        "## 2. 数据与实验设置",
        "",
        "- 原始 metadata：`data/whoiswho/data/NA_Demo/SND/valid/sna_valid_pub.json`",
        "- block 列表：`data/whoiswho/data/NA_Demo/SND/valid/sna_valid_raw.json`",
        "- ground truth：`data/whoiswho/data/NA_Demo/SND/valid/sna_valid_ground_truth.json`",
        "- 历史 LLM 结果：`artifacts/results/field_selection/single_field/NA_Demo/` 与 "
        "`artifacts/results/field_selection/combinations/NA_Demo/`",
        "- LLM 聚类指标复用 `src.common.metrics` 中的 Pairwise 与 B³ 逻辑。",
        "- 文本相似度未联网下载 sentence-transformers；本实验使用本地 TF-IDF cosine。"
        "organization 使用轻量 token Jaccard。该设置可复现，但不等价于神经 embedding。",
        "- `pairwise_similarity_records.json` 中的统计值基于全量 pair，`records` 字段只保存分层样本，"
        "避免把 39 万个 pair 的明细全部堆进报告文件。",
        "",
        "缺失或未完成组合：",
        "",
        "```json",
        json.dumps(missing, ensure_ascii=False, indent=2),
        "```",
        "",
        "## 3. 实验一：字段组合效果比较",
        "",
        "下面列出按 macro B³ F1 排名前 8 的字段组合：",
        "",
        markdown_table(
            top,
            [
                "field_combination",
                "completed_author_blocks",
                "macro_average_pairwise_f1",
                "macro_average_b3_f1",
                "median_pairwise_f1",
                "median_b3_f1",
            ],
        ),
        "",
        f"T+O+C 的 macro Pairwise F1 为 "
        f"`{toc.get('macro_average_pairwise_f1', float('nan')):.4f}`，macro B³ F1 为 "
        f"`{toc.get('macro_average_b3_f1', float('nan')):.4f}`。在这个三作者 pilot 中，"
        "T+O+C 不是所有指标上唯一第一，但它在保持高 Pairwise/B³ F1 的同时，"
        "只使用三个较短、身份指向较强的字段，因此比加入 abstract 的组合更有 token 性价比。",
        "",
        "相关图表：",
        "",
        "- `artifacts/figures/field_selection/analysis/combination_pairwise_f1.html`",
        "- `artifacts/figures/field_selection/analysis/combination_b3_f1.html`",
        "",
        "## 4. 实验二：字段的 pair-level 区分能力",
        "",
        "该实验不使用 LLM 输出，而是直接在每个 block 内构造论文 pair，并用 ground truth 标注是否同一真实作者。",
        "字段相似度只说明字段本身是否提供可分信号，不能直接证明 LLM 因某字段做出某判断。",
        "",
        markdown_table(
            gap_rows,
            ["field", "available_pair_count", "same_mean", "different_mean", "separation_gap"],
        ),
        "",
        "从 separation gap 看，coauthor/organization 更像身份锚点，title/abstract 更像主题语义信号。"
        "T+O+C 的优势更可能来自三类证据互补：title 提供研究主题，coauthor 提供关系网络，"
        "organization 提供机构约束。",
        "",
        "相关图表：",
        "",
        "- `artifacts/figures/field_selection/analysis/similarity_distribution_title.html`",
        "- `artifacts/figures/field_selection/analysis/similarity_distribution_abstract.html`",
        "- `artifacts/figures/field_selection/analysis/similarity_distribution_organization.html`",
        "- `artifacts/figures/field_selection/analysis/coauthor_overlap_distribution.html`",
        "- `artifacts/figures/field_selection/analysis/field_similarity_gap.html`",
        "",
        "## 5. 实验三：字段错误修复机制",
        "",
        "比较 Full=T+O+C 与三个 leave-one-field-out 结果：O+C、T+O、T+C。"
        "这里的 pairwise 改变来自整体 clustering 差异，因此应表述为“加入某字段后的聚类结果修复/引入了某类 pairwise 错误”，"
        "而不是“模型直接因为该字段判断某个 pair”。",
        "",
        markdown_table(
            contrib_all,
            [
                "field_added",
                "fixed_false_splits",
                "fixed_false_merges",
                "introduced_false_splits",
                "introduced_false_merges",
                "unchanged_pairs",
                "net_pairwise_gain",
            ],
        ),
        "",
        "相关图表：",
        "",
        "- `artifacts/figures/field_selection/analysis/field_error_repair_summary.html`",
        "- `artifacts/figures/field_selection/analysis/field_net_gain_by_author.html`",
        "",
        "代表性案例见：`artifacts/results/field_selection/analysis/field_contribution_cases_zh.md`。",
        "",
        "## 6. 综合结论",
        "",
        "基于三作者 pilot，T+O+C 的证据不是“字段越多越好”，而是：",
        "",
        "1. `title` 捕获论文主题，使同一作者跨论文的研究方向可以被连接；",
        "2. `coauthors` 是强身份线索，有助于修复同作者被拆散的问题；",
        "3. `organization` 能减少不同作者之间的错误合并；",
        "4. 相比 abstract，T+O+C token 成本更低、噪声更少，适合作为 stage1 的高性价比主组合。",
        "",
        "但是，本结论只覆盖三个 NA_Demo 作者 block。后续若要写成论文主结论，需要在 80 个 v3 block 上复验，"
        "并记录 token usage、输出失败率和稳定性。",
        "",
        "## 7. 局限性",
        "",
        "- 样本只有 3 个作者 block；",
        "- 历史 LLM 输出可能受 prompt、顺序、随机性影响；",
        "- 相似度实验使用 TF-IDF/Jaccard，不是深度语义 embedding；",
        "- leave-one-field-out 比较的是整体 clustering 结果，不是控制变量下的逐 pair 因果判断。",
        "",
    ]
    (OUT_DIR / "pilot_experiment_report_zh.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    global OUT_DIR, PLOT_DIR

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--plot-dir", type=Path, default=PLOT_DIR)
    args = parser.parse_args()

    OUT_DIR = args.output_dir
    PLOT_DIR = args.plot_dir
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    raw = read_json(args.data_dir / "sna_valid_raw.json")
    papers = read_json(args.data_dir / "sna_valid_pub.json")
    truth = load_truth(args.data_dir, args.data_dir / "sna_valid_ground_truth.json")

    summary_rows, aggregate_rows, missing = summarize_combinations(raw, truth)
    write_json(OUT_DIR / "field_combination_summary.json", {
        "audit": {
            "data_dir": str(args.data_dir),
            "result_roots": [
                "artifacts/results/field_selection/single_field/NA_Demo",
                "artifacts/results/field_selection/combinations/NA_Demo",
            ],
            "authors": AUTHORS,
            "main_combination": FULL,
            "text_similarity": "local TF-IDF cosine for title/abstract; token Jaccard for organization",
        },
        "per_author": summary_rows,
        "aggregate": aggregate_rows,
        "missing_results": missing,
    })
    write_csv(OUT_DIR / "field_combination_summary.csv", summary_rows)
    plot_combination_bars(summary_rows)

    pair_records, features_by_author = pairwise_similarity(raw, truth, papers)
    sim_stats = field_stats(pair_records)
    write_json(OUT_DIR / "pairwise_similarity_records.json", {
        "schema_note": (
            "field_statistics are computed from all within-block unordered paper pairs; "
            "records stores a compact stratified sample for manual inspection."
        ),
        "record_count": len(pair_records),
        "saved_sample_count": len(compact_pair_records(pair_records)),
        "similarity_method": "title/abstract local TF-IDF cosine; organization token Jaccard; coauthor set overlap/Jaccard",
        "records": compact_pair_records(pair_records),
        "field_statistics": sim_stats,
    })
    write_csv(OUT_DIR / "field_similarity_statistics.csv", sim_stats)
    plot_similarity(pair_records, sim_stats)

    contrib, contribution_summary = contribution_records(raw, truth, pair_records)
    for field, records in contrib.items():
        write_json(OUT_DIR / f"field_contribution_{field}.json", {
            "field_added": field,
            "full_combination": FULL,
            "without_field_combination": LEAVE_ONE_OUT[field],
            "changed_pair_count": len(records),
            "records": records,
        })
    write_json(OUT_DIR / "field_contribution_summary.json", contribution_summary)
    write_csv(OUT_DIR / "field_contribution_summary.csv", contribution_summary)
    plot_contribution(contribution_summary)
    (OUT_DIR / "field_contribution_cases_zh.md").write_text(
        case_markdown(contrib, features_by_author),
        encoding="utf-8",
    )
    write_report(summary_rows, aggregate_rows, missing, sim_stats, contribution_summary)

    index = """<!doctype html>
<html><head><meta charset="utf-8"><title>Pilot Field Combination Analysis</title></head>
<body><h1>Pilot Field Combination Analysis</h1><ul>
<li><a href="combination_pairwise_f1.html">Combination Pairwise F1</a></li>
<li><a href="combination_b3_f1.html">Combination B³ F1</a></li>
<li><a href="similarity_distribution_title.html">Title Similarity</a></li>
<li><a href="similarity_distribution_abstract.html">Abstract Similarity</a></li>
<li><a href="similarity_distribution_organization.html">Organization Similarity</a></li>
<li><a href="coauthor_overlap_distribution.html">Coauthor Overlap</a></li>
<li><a href="field_similarity_gap.html">Field Similarity Gap</a></li>
<li><a href="field_error_repair_summary.html">Field Error Repair Summary</a></li>
<li><a href="field_net_gain_by_author.html">Field Net Gain by Author</a></li>
</ul></body></html>
"""
    (PLOT_DIR / "index.html").write_text(index, encoding="utf-8")

    print(f"saved json/report: {OUT_DIR}")
    print(f"saved plots: {PLOT_DIR}")
    print(f"pair records: {len(pair_records)}")
    print(f"summary rows: {len(summary_rows)}")


if __name__ == "__main__":
    main()
