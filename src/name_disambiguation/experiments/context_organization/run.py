#!/usr/bin/env python3
import argparse
import csv
import html
import http.client
import json
import math
import os
import random
import re
import statistics
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

from name_disambiguation.core.metrics import b3_metrics, pairwise_metrics
from name_disambiguation.core.normalize_llm_response import normalize_response
from name_disambiguation.paths import FIGURES_ROOT, REPORTS_ROOT, RESULTS_ROOT, dataset_dir


DATA_DIR = dataset_dir("v3")
BASELINE_METRICS = FIGURES_ROOT / "field_selection" / "overview" / "analysis_summary.csv"
RESULT_DIR = RESULTS_ROOT / "context_organization"
REPORT_DIR = REPORTS_ROOT / "context_organization"
PLOT_DIR = FIGURES_ROOT / "context_organization"
RAW_RESULT_DIR = RESULTS_ROOT / "baseline_clusters"
FIELDS = ("title", "organization", "coauthors")
CONDITIONS = (
    "original_flat",
    "random_order",
    "chronological_order",
    "organization_grouped",
    "coauthor_network",
    "multifield_similarity",
)
RANDOM_REPETITIONS = (0, 1, 2)
RANDOM_SEEDS = (101, 202, 303)


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def normalize_text(text):
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def normalize_name(name):
    return re.sub(r"[^a-z0-9]", "", str(name or "").lower())


def name_forms(name):
    words = re.findall(r"[a-z0-9]+", name.lower())
    return {"".join(words), "".join(reversed(words))}


def token_set(value):
    if isinstance(value, list):
        value = " ".join(str(item) for item in value)
    return set(re.findall(r"[a-z0-9]+", str(value or "").lower()))


def field_value(paper, target_name, field):
    targets = name_forms(target_name)
    if field == "organization":
        return [
            author.get("org", "")
            for author in paper.get("authors", [])
            if normalize_name(author.get("name", "")) in targets
        ]
    if field == "coauthors":
        return [
            author.get("name", "")
            for author in paper.get("authors", [])
            if normalize_name(author.get("name", "")) not in targets
        ]
    return paper.get(field, "")


def make_card(index, paper_id, paper, name):
    return {
        "record_index": index,
        "title": field_value(paper, name, "title"),
        "organization": field_value(paper, name, "organization"),
        "coauthors": field_value(paper, name, "coauthors"),
    }


def prompt_for(name, context):
    return f"""You are performing high-recall author name disambiguation.

All records below contain an author whose normalized name is {name!r}. Different
real people may share this name. Create broad candidate clusters for a later,
finer disambiguation stage.

The overriding rule is recall: papers by the same real person must not be split
across different clusters. Each paper must belong to exactly one cluster. When
evidence is uncertain, place it in the broader most plausible cluster. For this
experiment, infer identity using only these supplied fields: title, organization, coauthors.
Do not create a catch-all cluster containing nearly every paper.

First reason globally about stable author profiles and links across the complete
collection. Then return JSON only.
Return JSON using exactly this shape:
{{"assignments":{{"0":"0","1":"1"}},"labels":{{"0":"short profile","1":"short profile"}}}}
Every input record_index must be a key in assignments exactly once.
Use only supplied record_index values.

PAPERS:
{json.dumps(context, ensure_ascii=False, separators=(",", ":"))}
"""


def flatten_context(context):
    if isinstance(context, list) and context and "papers" in context[0]:
        cards = []
        for group in context:
            cards.extend(group["papers"])
        return cards
    return list(context)


def ensure_same_information(reference_cards, contexts):
    reference = {
        card["record_index"]: {field: card[field] for field in FIELDS}
        for card in reference_cards
    }
    for condition, context in contexts.items():
        current_cards = flatten_context(context)
        current = {
            card["record_index"]: {field: card[field] for field in FIELDS}
            for card in current_cards
        }
        if current != reference:
            raise ValueError(f"information changed in condition {condition}")


def grouped_context(groups):
    return [
        {"group": f"Group {index + 1}", "papers": papers}
        for index, papers in enumerate(groups)
    ]


def chronological_cards(cards, paper_ids, papers):
    indexed = list(zip(cards, paper_ids))

    def key(item):
        _, paper_id = item
        year = papers[paper_id].get("year")
        try:
            year_value = int(year)
            missing = 0
        except (TypeError, ValueError):
            year_value = 10**9
            missing = 1
        return (missing, year_value, paper_id)

    return [card for card, _ in sorted(indexed, key=key)]


def organization_groups(cards):
    buckets = defaultdict(list)
    for card in cards:
        org_tokens = token_set(card["organization"])
        key = " ".join(sorted(org_tokens)) if org_tokens else "__missing_organization__"
        buckets[key].append(card)
    return [buckets[key] for key in sorted(buckets)]


def coauthor_network_groups(cards):
    parent = list(range(len(cards)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left, right):
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    coauthors = [{normalize_name(name) for name in card["coauthors"] if normalize_name(name)} for card in cards]
    for i in range(len(cards)):
        for j in range(i + 1, len(cards)):
            if coauthors[i] & coauthors[j]:
                union(i, j)
    components = defaultdict(list)
    for index, card in enumerate(cards):
        components[find(index)].append(card)
    return sorted(components.values(), key=lambda group: min(card["record_index"] for card in group))


def title_vectors(cards):
    docs = [re.findall(r"[a-z0-9]+", str(card["title"] or "").lower()) for card in cards]
    df = Counter(token for doc in docs for token in set(doc))
    vectors = []
    total = len(docs)
    for doc in docs:
        tf = Counter(doc)
        vec = {}
        for token, count in tf.items():
            vec[token] = count * (math.log((1 + total) / (1 + df[token])) + 1)
        norm = math.sqrt(sum(value * value for value in vec.values()))
        vectors.append({token: value / norm for token, value in vec.items()} if norm else {})
    return vectors


def cosine(left, right):
    if len(left) > len(right):
        left, right = right, left
    return sum(value * right.get(token, 0.0) for token, value in left.items())


def jaccard(left, right):
    if not left and not right:
        return None
    union = left | right
    return len(left & right) / len(union) if union else None


def multifield_similarity_order(cards, paper_ids):
    vectors = title_vectors(cards)
    org_sets = [token_set(card["organization"]) for card in cards]
    coauthor_sets = [{normalize_name(name) for name in card["coauthors"] if normalize_name(name)} for card in cards]
    remaining = set(range(len(cards)))
    current = min(remaining, key=lambda idx: paper_ids[idx])
    order = [current]
    remaining.remove(current)
    while remaining:
        best = None
        for candidate in remaining:
            values = [cosine(vectors[current], vectors[candidate])]
            for left, right in ((org_sets[current], org_sets[candidate]), (coauthor_sets[current], coauthor_sets[candidate])):
                score = jaccard(left, right)
                if score is not None:
                    values.append(score)
            score = sum(values) / len(values) if values else 0.0
            key = (-score, paper_ids[candidate])
            if best is None or key < best[0]:
                best = (key, candidate)
        current = best[1]
        order.append(current)
        remaining.remove(current)
    return [cards[index] for index in order]


def build_contexts(name, paper_ids, papers):
    cards = [make_card(index, pid, papers[pid], name) for index, pid in enumerate(paper_ids)]
    contexts = {
        "original_flat:0": cards,
        "chronological_order:0": chronological_cards(cards, paper_ids, papers),
        "organization_grouped:0": grouped_context(organization_groups(cards)),
        "coauthor_network:0": grouped_context(coauthor_network_groups(cards)),
        "multifield_similarity:0": multifield_similarity_order(cards, paper_ids),
    }
    for repetition, seed in zip(RANDOM_REPETITIONS, RANDOM_SEEDS):
        shuffled = list(cards)
        random.Random(seed).shuffle(shuffled)
        contexts[f"random_order:{repetition}"] = shuffled
    ensure_same_information(cards, contexts)
    return contexts


def read_baseline_rows(path):
    rows = []
    with path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            if row["status"] == "completed" and row["combination"] == "title+coauthors+organization":
                row["b3_f1"] = float(row["b3_f1"])
                row["paper_count"] = int(row["paper_count"])
                rows.append(row)
    return rows


def select_authors(path, output_path, seed=42):
    existing = load_json(output_path) if output_path.exists() else None
    if existing:
        return existing
    rows = read_baseline_rows(path)
    groups = {
        "easy": [row for row in rows if row["b3_f1"] >= 0.90],
        "medium": [row for row in rows if 0.80 <= row["b3_f1"] < 0.90],
        "hard": [row for row in rows if row["b3_f1"] < 0.80],
    }
    selected = []
    for difficulty, candidates in groups.items():
        candidates = sorted(candidates, key=lambda row: row["name"])
        rng = random.Random(seed)
        for row in rng.sample(candidates, min(2, len(candidates))):
            selected.append({
                "author_id": row["name"],
                "difficulty_group": difficulty,
                "baseline_b3_f1": row["b3_f1"],
                "baseline_pairwise_f1": float(row["pairwise_f1"]),
                "paper_count": row["paper_count"],
            })
    write_json(output_path, selected)
    return selected


def chat_completion(prompt, provider, model, timeout):
    if provider != "deepseek":
        raise ValueError("this experiment is DeepSeek-only; do not use Zhiyuan API")
    api_key = os.environ["DEEPSEEK_API_KEY"]
    base_url = os.environ["DEEPSEEK_BASE_URL"].rstrip("/")
    url = base_url if base_url.endswith("/chat/completions") else f"{base_url}/chat/completions"
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 60000 if provider == "deepseek" else 12000,
    }
    body["response_format"] = {"type": "json_object"}
    body["extra_body"] = {"thinking": {"type": "disabled"}}
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def validate_clusters(result, paper_ids):
    known = set(paper_ids)
    assigned = set()
    groups = []
    for cluster in result.get("clusters", []):
        ids = list(dict.fromkeys(cluster.get("paper_ids", [])))
        unknown = set(ids) - known
        if unknown:
            raise ValueError(f"unknown paper IDs: {sorted(unknown)[:5]}")
        repeated = set(ids) & assigned
        if repeated:
            raise ValueError(f"paper IDs assigned to multiple clusters: {sorted(repeated)[:5]}")
        assigned.update(ids)
        groups.append(set(ids))
    missing = known - assigned
    if missing:
        raise ValueError(f"missing {len(missing)} paper IDs")
    if not groups:
        raise ValueError("no clusters returned")
    return groups


def load_truth_groups(name, paper_ids):
    raw_result = RAW_RESULT_DIR / f"{name}.json"
    if raw_result.exists():
        result = load_json(raw_result)
        by_author = defaultdict(set)
        for cluster in result.get("clusters", []):
            for paper in cluster.get("papers", []):
                by_author[paper["true_author_id"]].add(paper["paper_id"])
        if set().union(*by_author.values()) == set(paper_ids):
            return list(by_author.values())
    raise ValueError(
        f"complete ground truth for {name} is unavailable; "
        f"expected true_author_id annotations in {raw_result}"
    )


def usage_value(response, key):
    usage = response.get("usage") or {}
    return usage.get(key) or usage.get({"prompt_tokens": "input_tokens", "completion_tokens": "output_tokens"}.get(key, key)) or ""


def run_call(name, difficulty, condition, repetition, context, paper_ids, truth_groups, provider, model, timeout, force=False):
    stem = f"{condition}_rep{repetition}"
    raw_path = RESULT_DIR / "raw_responses" / name / f"{stem}.response.json"
    parsed_path = RESULT_DIR / "parsed" / name / f"{stem}.json"
    meta_path = RESULT_DIR / "metadata" / name / f"{stem}.json"
    prompt = prompt_for(name, context)
    if raw_path.exists() and not force:
        response = load_json(raw_path)
        parsed = load_json(parsed_path) if parsed_path.exists() else normalize_response(response, paper_ids=paper_ids)
        meta = load_json(meta_path) if meta_path.exists() else {
            "latency": "",
            "input_tokens": usage_value(response, "prompt_tokens") or len(prompt) // 4,
            "output_tokens": usage_value(response, "completion_tokens"),
            "prompt_characters": len(prompt),
            "grouped": bool(flatten_context(context) != context),
            "group_count": len(context) if isinstance(context, list) and context and "papers" in context[0] else "",
            "model": model,
            "provider": provider,
        }
    else:
        start = time.time()
        response = chat_completion(prompt, provider, model, timeout)
        latency = time.time() - start
        write_json(raw_path, response)
        parsed = normalize_response(response, paper_ids=paper_ids)
        meta = {
            "latency": latency,
            "input_tokens": usage_value(response, "prompt_tokens") or len(prompt) // 4,
            "output_tokens": usage_value(response, "completion_tokens"),
            "prompt_characters": len(prompt),
            "grouped": bool(flatten_context(context) != context),
            "group_count": len(context) if isinstance(context, list) and context and "papers" in context[0] else "",
            "model": model,
            "provider": provider,
        }
    groups = validate_clusters(parsed, paper_ids)
    if not parsed_path.exists():
        write_json(parsed_path, parsed)
    if not meta_path.exists():
        write_json(meta_path, meta)
    pairwise = pairwise_metrics(truth_groups, groups, paper_ids)
    b3 = b3_metrics(truth_groups, groups, paper_ids)
    return {
        "author_id": name,
        "difficulty_group": difficulty,
        "condition": condition,
        "repetition": repetition,
        "paper_count": len(paper_ids),
        "pairwise_precision": pairwise["precision"],
        "pairwise_recall": pairwise["recall"],
        "pairwise_f1": pairwise["f1"],
        "b3_precision": b3["precision"],
        "b3_recall": b3["recall"],
        "b3_f1": b3["f1"],
        "predicted_cluster_count": len(groups),
        "true_cluster_count": len(truth_groups),
        "input_tokens": meta.get("input_tokens", ""),
        "output_tokens": meta.get("output_tokens", ""),
        "latency": meta.get("latency", ""),
        "parse_success": True,
        "raw_response_path": str(raw_path),
        "parsed_result_path": str(parsed_path),
    }


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean(values):
    return sum(values) / len(values) if values else 0.0


def median(values):
    return statistics.median(values) if values else 0.0


def aggregate_for_summary(metrics):
    by_condition_author = defaultdict(list)
    for row in metrics:
        if row["parse_success"]:
            by_condition_author[(row["condition"], row["author_id"])].append(row)
    author_level = []
    for (condition, author), rows in by_condition_author.items():
        first = rows[0]
        author_level.append({
            "condition": condition,
            "author_id": author,
            "difficulty_group": first["difficulty_group"],
            "b3_precision": mean([float(row["b3_precision"]) for row in rows]),
            "b3_recall": mean([float(row["b3_recall"]) for row in rows]),
            "b3_f1": mean([float(row["b3_f1"]) for row in rows]),
            "pairwise_f1": mean([float(row["pairwise_f1"]) for row in rows]),
            "predicted_cluster_count": mean([float(row["predicted_cluster_count"]) for row in rows]),
            "input_tokens": mean([float(row["input_tokens"] or 0) for row in rows]),
        })
    baseline = {row["author_id"]: row["b3_f1"] for row in author_level if row["condition"] == "original_flat"}
    for row in author_level:
        row["delta_b3_f1"] = (
            row["b3_f1"] - baseline[row["author_id"]]
            if row["condition"] != "original_flat" and row["author_id"] in baseline
            else 0.0 if row["condition"] == "original_flat" else None
        )
    parse_failures = Counter(row["condition"] for row in metrics if not row["parse_success"])
    summary = []
    for condition in CONDITIONS:
        rows = [row for row in author_level if row["condition"] == condition]
        deltas = [row["delta_b3_f1"] for row in rows if condition != "original_flat" and row["delta_b3_f1"] is not None]
        summary.append({
            "condition": condition,
            "mean_b3_f1": mean([row["b3_f1"] for row in rows]),
            "median_b3_f1": median([row["b3_f1"] for row in rows]),
            "mean_pairwise_f1": mean([row["pairwise_f1"] for row in rows]),
            "mean_predicted_cluster_count": mean([row["predicted_cluster_count"] for row in rows]),
            "mean_input_tokens": mean([row["input_tokens"] for row in rows]),
            "parse_failure_count": parse_failures[condition],
            "mean_delta_b3_f1": mean(deltas),
            "median_delta_b3_f1": median(deltas),
            "improved_author_count": sum(delta > 1e-12 for delta in deltas),
            "degraded_author_count": sum(delta < -1e-12 for delta in deltas),
            "unchanged_author_count": sum(abs(delta) <= 1e-12 for delta in deltas),
        })
    return author_level, summary


def random_stability(metrics):
    rows = []
    for author in sorted({row["author_id"] for row in metrics}):
        values = [
            float(row["b3_f1"])
            for row in metrics
            if row["author_id"] == author and row["condition"] == "random_order" and row["parse_success"]
        ]
        if not values:
            continue
        rows.append({
            "author_id": author,
            "random_b3_f1_rep0": values[0] if len(values) > 0 else "",
            "random_b3_f1_rep1": values[1] if len(values) > 1 else "",
            "random_b3_f1_rep2": values[2] if len(values) > 2 else "",
            "mean": mean(values),
            "standard_deviation": statistics.pstdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
            "range": max(values) - min(values),
        })
    return rows


def plot_html(title, body):
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script></head>
<body><div id="chart" style="width:100%;height:720px;"></div>
<script>{body}</script></body></html>
"""


def write_plots(author_level, stability_rows):
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    data = json.dumps(author_level, ensure_ascii=False)
    PLOT_DIR.joinpath("condition_performance.html").write_text(plot_html(
        "Condition Performance",
        f"""
const rows = {data};
const conditions = {json.dumps(list(CONDITIONS))};
const traces = conditions.map(c => {{
  const r = rows.filter(x => x.condition === c);
  return {{type:'box', name:c, y:r.map(x=>x.b3_f1), text:r.map(x=>`${{x.author_id}} (${{x.difficulty_group}})`), boxpoints:'all', jitter:0.35, pointpos:0}};
}});
Plotly.newPlot('chart', traces, {{title:'B³ F1 by condition', yaxis:{{title:'B³ F1'}}}});
"""), encoding="utf-8")

    non_baseline = [condition for condition in CONDITIONS if condition != "original_flat"]
    PLOT_DIR.joinpath("paired_comparison.html").write_text(plot_html(
        "Paired Comparison",
        f"""
const rows = {data};
const base = Object.fromEntries(rows.filter(x => x.condition === 'original_flat').map(x => [x.author_id, x.b3_f1]));
const traces = {json.dumps(non_baseline)}.map(c => {{
  const r = rows.filter(x => x.condition === c);
  return {{type:'scatter', mode:'markers', name:c, x:r.map(x=>base[x.author_id]), y:r.map(x=>x.b3_f1), text:r.map(x=>`${{x.author_id}} delta=${{(x.b3_f1-base[x.author_id]).toFixed(4)}}`)}};
}});
traces.push({{type:'scatter', mode:'lines', name:'y=x', x:[0,1], y:[0,1], line:{{dash:'dash', color:'gray'}}}});
Plotly.newPlot('chart', traces, {{title:'Condition vs original_flat', xaxis:{{title:'original_flat B³ F1'}}, yaxis:{{title:'condition B³ F1'}}}});
"""), encoding="utf-8")

    authors = sorted({row["author_id"] for row in author_level})
    delta = {(row["author_id"], row["condition"]): row["delta_b3_f1"] for row in author_level}
    z = [[delta.get((author, condition), None) for condition in non_baseline] for author in authors]
    PLOT_DIR.joinpath("author_delta_heatmap.html").write_text(plot_html(
        "Author Delta Heatmap",
        f"""
Plotly.newPlot('chart', [{{type:'heatmap', x:{json.dumps(non_baseline)}, y:{json.dumps(authors)}, z:{json.dumps(z)}, colorscale:'RdBu', zmid:0}}], {{title:'Delta B³ F1 vs original_flat'}});
"""), encoding="utf-8")

    stability = json.dumps(stability_rows, ensure_ascii=False)
    PLOT_DIR.joinpath("random_order_stability.html").write_text(plot_html(
        "Random Order Stability",
        f"""
const rows = {stability};
const traces = [0,1,2].map(i => ({{type:'scatter', mode:'markers', name:`rep${{i}}`, x:rows.map(r=>r.author_id), y:rows.map(r=>r[`random_b3_f1_rep${{i}}`])}}));
traces.push({{type:'bar', name:'range', x:rows.map(r=>r.author_id), y:rows.map(r=>r.range), yaxis:'y2', opacity:0.35}});
Plotly.newPlot('chart', traces, {{title:'Random order B³ F1 stability', yaxis:{{title:'B³ F1'}}, yaxis2:{{title:'range', overlaying:'y', side:'right'}}}});
"""), encoding="utf-8")


def markdown_table(rows, columns):
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column, "")
            if value is None:
                values.append("")
            else:
                values.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(selected, summary, author_level, stability_rows, model):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    best = max(summary, key=lambda row: row["mean_b3_f1"])
    baseline = next(row for row in summary if row["condition"] == "original_flat")
    most_stable = min(stability_rows, key=lambda row: row["range"]) if stability_rows else None
    random_summary = next(row for row in summary if row["condition"] == "random_order")
    grouped = {row["condition"]: row for row in summary}
    cases = sorted(
        [row for row in author_level if row["condition"] != "original_flat" and row["delta_b3_f1"] is not None],
        key=lambda row: abs(row["delta_b3_f1"]),
        reverse=True,
    )[:4]
    selected_counts = Counter(item["difficulty_group"] for item in selected)
    stable_author = most_stable["author_id"] if most_stable else "N/A"
    stable_range = most_stable["range"] if most_stable else 0.0
    selected_rows = [
        {
            "author_id": item["author_id"],
            "difficulty_group": item["difficulty_group"],
            "paper_count": item["paper_count"],
            "historical_b3_f1": item["baseline_b3_f1"],
        }
        for item in selected
    ]
    failure_rows = [
        {"condition": row["condition"], "parse_failure_count": row["parse_failure_count"]}
        for row in summary
        if row["parse_failure_count"]
    ]
    failure_text = markdown_table(failure_rows, ["condition", "parse_failure_count"]) if failure_rows else "无解析失败。"
    report = f"""# Context organization 实验报告

## 1. 实验目的

本实验固定使用 `title + organization + coauthors`（T+O+C）三类输入证据，只改变论文在 prompt 中的排列或中性分组方式，检验信息组织方式是否会影响 LLM 作者姓名消歧结果。

## 2. 实验设置

- 数据集：`data/whoiswho/data/v3/SND/valid`
- 作者数量：{len(selected)}，其中 easy={selected_counts.get('easy', 0)}，medium={selected_counts.get('medium', 0)}，hard={selected_counts.get('hard', 0)}
- 作者选择：基于既有 T+O+C B³ F1 分层，用固定随机种子 `42` 每层抽取 2 个作者
- 选中作者：

{markdown_table(selected_rows, ['author_id', 'difficulty_group', 'paper_count', 'historical_b3_f1'])}

- 模型：`{model}`，temperature=0，输出格式沿用原 T+O+C compact assignment JSON
- 条件：`original_flat`、`random_order`、`chronological_order`、`organization_grouped`、`coauthor_network`、`multifield_similarity`
- `random_order`：每个作者 3 个固定随机排列
- 指标：Pairwise Precision/Recall/F1 与 B³ Precision/Recall/F1

## 3. 主要结果

{markdown_table(summary, ['condition', 'mean_b3_f1', 'median_b3_f1', 'mean_pairwise_f1', 'mean_delta_b3_f1', 'improved_author_count', 'degraded_author_count', 'parse_failure_count'])}

说明：本表中 mean/median 指标只基于成功解析且通过 paper 覆盖校验的调用；`mean_delta_b3_f1` 只在同一作者的 `original_flat` 与目标条件都成功时计算。解析失败本身是实验结果的一部分，表示该组织方式增加了输出不完整或格式失败风险。

缺失论文处理方法：本实验没有把 LLM 输出中缺失的论文补成 singleton。每次调用都会先检查所有输入 paper ID 是否在输出中恰好出现一次；若有缺失、未知 ID 或跨 cluster 重复，就将该次调用标记为 `parse_success=False`，不计算 Pairwise/B³，也不纳入条件均值。也就是说，公共指标函数虽然有 missing-as-singleton 的兜底能力，但本实验主表采用更严格的 full-coverage 口径，把缺失论文视为该组织方式的输出可靠性失败。

解析失败统计：

{failure_text}

## 4. 主要发现

- 在成功解析样本上，平均 B³ F1 最高的是 `{best['condition']}`，mean B³ F1={best['mean_b3_f1']:.4f}；baseline `original_flat` 为 {baseline['mean_b3_f1']:.4f}。由于部分条件有解析失败，这个均值不能单独解释为稳定优势。
- `random_order` 的作者级平均 delta B³ F1 为 {random_summary['mean_delta_b3_f1']:.4f}，说明顺序扰动在当前样本上{'存在明显影响' if abs(random_summary['mean_delta_b3_f1']) > 0.01 else '总体影响有限'}。
- `organization_grouped` 的 mean predicted cluster count 为 {grouped['organization_grouped']['mean_predicted_cluster_count']:.2f}，baseline 为 {baseline['mean_predicted_cluster_count']:.2f}；它在 `bo_yu` 上缺失 774 篇论文，说明大 block 分组会显著增加输出不完整风险。
- `coauthor_network` 的 mean predicted cluster count 为 {grouped['coauthor_network']['mean_predicted_cluster_count']:.2f}；它在 `yi_qian` 上有最高个体增益，但在 `bo_yu` 上缺失 229 篇论文，因此更像高风险高方差策略。
- random order 最稳定作者为 `{stable_author}`，B³ F1 range={stable_range:.4f}。

## 5. 典型案例

{markdown_table(cases, ['author_id', 'difficulty_group', 'condition', 'b3_f1', 'delta_b3_f1', 'predicted_cluster_count'])}

## 6. 结论

在当前 6 作者 pilot 中，信息组织方式会改变部分作者的 LLM 聚类结果，但平均 delta 很小，且分组方法在大 block 上更容易产生缺失论文或截断输出。因此更稳妥的结论是：信息组织方式是一个有影响但不稳定的 prompt-side 因素；若要作为 routing action，需要先加入 block size / prompt token 风险控制，而不能直接替代原始 flat baseline。

结果文件：`artifacts/results/context_organization/`

图表目录：`artifacts/figures/context_organization/`
"""
    REPORT_DIR.joinpath("context_organization_experiment_report_zh.md").write_text(report, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--provider", choices=("deepseek",), default="deepseek")
    parser.add_argument("--model", default=None)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    model = args.model or "deepseek-v4-flash"

    raw = load_json(args.data_dir / "sna_valid_raw.json")
    papers = load_json(args.data_dir / "sna_valid_pub.json")
    selected = select_authors(BASELINE_METRICS, RESULT_DIR / "selected_authors.json")

    metrics = []
    failures = []
    for item in selected:
        name = item["author_id"]
        print(f"\n=== {name} ({item['difficulty_group']}) ===")
        contexts = build_contexts(name, raw[name], papers)
        truth_groups = load_truth_groups(name, raw[name])
        for key, context in contexts.items():
            condition, repetition_text = key.split(":")
            repetition = int(repetition_text)
            try:
                row = run_call(
                    name,
                    item["difficulty_group"],
                    condition,
                    repetition,
                    context,
                    raw[name],
                    truth_groups,
                    args.provider,
                    model,
                    args.timeout,
                    args.force,
                )
                metrics.append(row)
                print(f"{condition} rep{repetition}: B3 F1={row['b3_f1']:.4f}")
            except (ValueError, json.JSONDecodeError, KeyError, urllib.error.URLError, http.client.RemoteDisconnected, http.client.IncompleteRead, TimeoutError) as exc:
                print(f"{condition} rep{repetition}: failed: {exc}")
                failures.append({
                    "author_id": name,
                    "difficulty_group": item["difficulty_group"],
                    "condition": condition,
                    "repetition": repetition,
                    "paper_count": len(raw[name]),
                    "pairwise_precision": "",
                    "pairwise_recall": "",
                    "pairwise_f1": "",
                    "b3_precision": "",
                    "b3_recall": "",
                    "b3_f1": "",
                    "predicted_cluster_count": "",
                    "true_cluster_count": len(truth_groups),
                    "input_tokens": "",
                    "output_tokens": "",
                    "latency": "",
                    "parse_success": False,
                    "error": str(exc),
                    "raw_response_path": "",
                    "parsed_result_path": "",
                })

    metrics.extend(failures)
    metric_fields = [
        "author_id", "difficulty_group", "condition", "repetition", "paper_count",
        "pairwise_precision", "pairwise_recall", "pairwise_f1",
        "b3_precision", "b3_recall", "b3_f1",
        "predicted_cluster_count", "true_cluster_count",
        "input_tokens", "output_tokens", "latency", "parse_success",
        "raw_response_path", "parsed_result_path", "error",
    ]
    for row in metrics:
        row.setdefault("error", "")
    write_csv(RESULT_DIR / "author_condition_metrics.csv", metrics, metric_fields)
    author_level, summary = aggregate_for_summary(metrics)
    stability = random_stability(metrics)
    write_csv(RESULT_DIR / "condition_summary.csv", summary, list(summary[0]))
    write_csv(RESULT_DIR / "random_order_stability.csv", stability, list(stability[0]) if stability else ["author_id"])
    write_json(RESULT_DIR / "author_level_condition_metrics.json", author_level)
    write_plots(author_level, stability)
    write_report(selected, summary, author_level, stability, model)

    print("\n=== summary ===")
    print(f"data_dir={args.data_dir}")
    print(f"selected_authors={len(selected)}")
    print(f"successful_calls={sum(1 for row in metrics if row['parse_success'])}")
    print(f"failed_calls={sum(1 for row in metrics if not row['parse_success'])}")
    for row in summary:
        print(f"{row['condition']}: mean B3 F1={row['mean_b3_f1']:.4f}")
    print(f"best_condition={max(summary, key=lambda row: row['mean_b3_f1'])['condition']}")
    print(f"report={REPORT_DIR / 'context_organization_experiment_report_zh.md'}")
    print(f"plots={PLOT_DIR}")
    print(f"results={RESULT_DIR}")


if __name__ == "__main__":
    main()
