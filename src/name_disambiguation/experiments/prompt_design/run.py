#!/usr/bin/env python3
import argparse
import csv
import html
import http.client
import json
import os
import random
import re
import signal
import statistics
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

from name_disambiguation.core.metrics import b3_metrics, pairwise_metrics
from name_disambiguation.paths import FIGURES_ROOT, REPORTS_ROOT, RESULTS_ROOT, dataset_dir


DATA_DIR = dataset_dir("v3")
BASELINE_METRICS = FIGURES_ROOT / "field_selection" / "overview" / "analysis_summary.csv"
RESULT_DIR = RESULTS_ROOT / "prompt_design"
REPORT_DIR = REPORTS_ROOT / "prompt_design"
PLOT_DIR = FIGURES_ROOT / "prompt_design"
RAW_RESULT_DIR = RESULTS_ROOT / "baseline_clusters"
CONDITIONS = ("minimal", "task_definition", "evidence_guidance", "error_aware", "analyze_then_cluster")
SEED = 42


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def normalize_name(name):
    return re.sub(r"[^a-z0-9]", "", str(name or "").lower())


def name_forms(name):
    words = re.findall(r"[a-z0-9]+", name.lower())
    return {"".join(words), "".join(reversed(words))}


def field_value(paper, target_name, field):
    targets = name_forms(target_name)
    if field == "organization":
        return [a.get("org", "") for a in paper.get("authors", []) if normalize_name(a.get("name", "")) in targets]
    if field == "coauthors":
        return [a.get("name", "") for a in paper.get("authors", []) if normalize_name(a.get("name", "")) not in targets]
    return paper.get(field, "")


def make_cards(name, paper_ids, papers):
    return [{
        "paper_id": pid,
        "display_id": f"P{i + 1}",
        "title": field_value(papers[pid], name, "title"),
        "organization": field_value(papers[pid], name, "organization"),
        "coauthors": field_value(papers[pid], name, "coauthors"),
    } for i, pid in enumerate(paper_ids)]


def context_text(cards):
    blocks = []
    for card in cards:
        blocks.append("\n".join([
            f"Paper ID: {card['display_id']}",
            f"Title: {card['title'] or ''}",
            "Organization: " + "; ".join(str(x) for x in card["organization"] if x),
            "Coauthors: " + "; ".join(str(x) for x in card["coauthors"] if x),
        ]))
    return "\n\n".join(blocks)


def instruction_for(condition):
    minimal = """The following publications share the same or a similar author name.
Group the publications by their actual authors using the provided information."""
    task = minimal + """

This is an author name disambiguation task. Publications with the same displayed
name may belong to different people. At the same time, one person may change
organization, research topic, or collaborators over time. Use all available
evidence jointly and do not make decisions based on only one field."""
    evidence = task + """

Use the fields as follows:
- Titles provide topic evidence, but one author may change topics and different authors may work on similar topics.
- Organizations provide affiliation evidence, but authors may move and the same organization may appear in different forms.
- Coauthors provide relationship evidence, but collaboration networks may also change.
Combine these signals instead of treating any single field as decisive."""
    error_aware = task + """

Avoid common mistakes:
- Do not split publications only because organizations or topics differ.
- Do not merge publications only because organizations or topics are similar.
- Shared coauthors are useful evidence, but are not absolute proof.
- Avoid both excessive splitting of one person and excessive merging of different people."""
    analyze = evidence + """

First examine supporting and conflicting evidence and check global consistency.
Then return only the final JSON clustering result."""
    return {
        "minimal": minimal,
        "task_definition": task,
        "evidence_guidance": evidence,
        "error_aware": error_aware,
        "analyze_then_cluster": analyze,
    }[condition]


def output_requirement():
    return """Return JSON only using exactly this shape:
{"assignments":{"P1":"0","P2":"1","P3":"0"},"labels":{"0":"short profile","1":"short profile"}}
Every supplied Paper ID must be a key in assignments exactly once. Use only supplied Paper IDs."""


def build_prompt(cards, condition):
    return "\n\n".join([
        "You are performing author name disambiguation.",
        instruction_for(condition),
        "PAPERS:",
        context_text(cards),
        output_requirement(),
    ])


def check_prompt_invariance(cards, prompts):
    fixed_context = context_text(cards)
    fixed_output = output_requirement()
    expected_ids = [card["display_id"] for card in cards]
    for condition, prompt in prompts.items():
        if fixed_context not in prompt:
            raise ValueError(f"context changed for {condition}")
        if fixed_output not in prompt:
            raise ValueError(f"output schema changed for {condition}")
        ids = [card["display_id"] for card in cards]
        if ids != expected_ids:
            raise ValueError(f"paper order changed for {condition}")


def read_baseline_rows():
    rows = []
    with BASELINE_METRICS.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["status"] == "completed" and row["combination"] == "title+coauthors+organization":
                for key in ("paper_count",):
                    row[key] = int(row[key])
                for key in ("b3_f1", "organization_coverage", "coauthor_coverage"):
                    row[key] = float(row[key])
                rows.append(row)
    return rows


def choose_one(rows, difficulty, median_papers, rng):
    if difficulty == "easy":
        pool = [r for r in rows if r["b3_f1"] >= 0.90]
    elif difficulty == "medium":
        pool = [r for r in rows if 0.80 <= r["b3_f1"] < 0.90]
    else:
        pool = [r for r in rows if 0.60 <= r["b3_f1"] < 0.80]
    preferred = [r for r in pool if r["paper_count"] <= median_papers and r["organization_coverage"] >= 0.7 and r["coauthor_coverage"] >= 0.9]
    pool = sorted(preferred or pool, key=lambda r: (r["paper_count"], r["name"]))
    return rng.choice(pool[:min(10, len(pool))])


def select_authors():
    path = RESULT_DIR / "selected_authors.json"
    if path.exists():
        return load_json(path)
    rows = read_baseline_rows()
    median_papers = statistics.median(r["paper_count"] for r in rows)
    rng = random.Random(SEED)
    selected = []
    for difficulty in ("easy", "medium", "hard"):
        row = choose_one(rows, difficulty, median_papers, rng)
        selected.append({
            "author_id": row["name"],
            "difficulty_group": difficulty,
            "baseline_b3_f1": row["b3_f1"],
            "paper_count": row["paper_count"],
            "organization_coverage": row["organization_coverage"],
            "coauthor_coverage": row["coauthor_coverage"],
        })
    write_json(path, selected)
    return selected


def with_deadline(seconds, func, *args):
    def handler(signum, frame):
        raise TimeoutError(f"API call exceeded {seconds} seconds")
    previous = signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        return func(*args)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def chat_completion(prompt, model, timeout):
    base_url = os.environ["DEEPSEEK_BASE_URL"].rstrip("/")
    url = base_url if base_url.endswith("/chat/completions") else f"{base_url}/chat/completions"
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 60000,
        "response_format": {"type": "json_object"},
        "extra_body": {"thinking": {"type": "disabled"}},
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": f"Bearer {os.environ['DEEPSEEK_API_KEY']}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def parse_response(response, cards):
    content = response["choices"][0]["message"].get("content", "").strip()
    data = json.loads(content)
    id_map = {c["display_id"]: c["paper_id"] for c in cards}
    if isinstance(data.get("assignments"), dict):
        labels = data.get("labels") or {}
        grouped = {}
        for display_id, cluster_id in data["assignments"].items():
            grouped.setdefault(str(cluster_id), []).append(id_map.get(display_id, display_id))
        return {
            "clusters": [
                {"label": labels.get(cluster_id, f"cluster {cluster_id}"), "paper_ids": ids}
                for cluster_id, ids in grouped.items()
            ]
        }

    clusters = data.get("clusters")
    if not isinstance(clusters, list):
        raise ValueError("response JSON has neither assignments nor clusters")
    parsed = []
    for i, cluster in enumerate(clusters):
        if isinstance(cluster, dict):
            ids = cluster.get("paper_ids", [])
            label = cluster.get("label", f"cluster {i}")
        else:
            ids = cluster
            label = f"cluster {i}"
        parsed.append({"label": label, "paper_ids": [id_map.get(pid, pid) for pid in ids]})
    return {"clusters": parsed}


def validate(parsed, paper_ids):
    known = set(paper_ids)
    assigned = []
    for cluster in parsed["clusters"]:
        ids = list(dict.fromkeys(cluster["paper_ids"]))
        cluster["paper_ids"] = ids
        assigned.extend(ids)
    counts = Counter(assigned)
    duplicate = [pid for pid, count in counts.items() if count > 1]
    unknown = sorted(set(assigned) - known)
    missing = sorted(known - set(assigned))
    if duplicate:
        raise ValueError(f"duplicate paper IDs: {duplicate[:5]}")
    if unknown:
        raise ValueError(f"unknown paper IDs: {unknown[:5]}")
    if missing:
        raise ValueError(f"missing {len(missing)} paper IDs")


def load_truth_groups(name, paper_ids):
    raw_result = RAW_RESULT_DIR / f"{name}.json"
    if raw_result.exists():
        result = load_json(raw_result)
        by_author = defaultdict(set)
        for cluster in result.get("clusters", []):
            for paper in cluster.get("papers", []):
                by_author[paper["true_author_id"]].add(paper["paper_id"])
        if by_author and set().union(*by_author.values()) == set(paper_ids):
            return list(by_author.values())
    raise ValueError(f"complete true_author_id annotations unavailable for {name}: {raw_result}")


def usage(response, key):
    u = response.get("usage") or {}
    return u.get(key) or u.get({"prompt_tokens": "input_tokens", "completion_tokens": "output_tokens"}.get(key, key)) or 0


def run_call(item, condition, cards, paper_ids, truth, model, timeout, force):
    name = item["author_id"]
    prompt = build_prompt(cards, condition)
    prompt_path = RESULT_DIR / "prompts" / name / f"{condition}.txt"
    raw_path = RESULT_DIR / "raw_responses" / name / f"{condition}.response.json"
    parsed_path = RESULT_DIR / "parsed" / name / f"{condition}.json"
    meta_path = RESULT_DIR / "metadata" / name / f"{condition}.json"
    write_text(prompt_path, prompt)

    if raw_path.exists() and parsed_path.exists() and meta_path.exists() and not force:
        response = load_json(raw_path)
        parsed = load_json(parsed_path)
        meta = load_json(meta_path)
    elif raw_path.exists() and not force:
        response = load_json(raw_path)
        parsed = parse_response(response, cards)
        validate(parsed, paper_ids)
        meta = {"input_tokens": usage(response, "prompt_tokens") or len(prompt) // 4, "output_tokens": usage(response, "completion_tokens"), "total_tokens": usage(response, "total_tokens"), "latency": "", "retry_count": 0}
        write_json(parsed_path, parsed)
        write_json(meta_path, meta)
    else:
        start = time.time()
        response = with_deadline(timeout, chat_completion, prompt, model, timeout)
        latency = time.time() - start
        write_json(raw_path, response)
        parsed = parse_response(response, cards)
        validate(parsed, paper_ids)
        meta = {"input_tokens": usage(response, "prompt_tokens") or len(prompt) // 4, "output_tokens": usage(response, "completion_tokens"), "total_tokens": usage(response, "total_tokens"), "latency": latency, "retry_count": 0}
        write_json(parsed_path, parsed)
        write_json(meta_path, meta)

    groups = [set(c["paper_ids"]) for c in parsed["clusters"]]
    truth_groups = truth
    pairwise = pairwise_metrics(truth_groups, groups, paper_ids)
    b3 = b3_metrics(truth_groups, groups, paper_ids)
    return {
        "author_id": name,
        "difficulty_group": item["difficulty_group"],
        "condition": condition,
        "paper_count": len(paper_ids),
        "pairwise_precision": pairwise["precision"],
        "pairwise_recall": pairwise["recall"],
        "pairwise_f1": pairwise["f1"],
        "b3_precision": b3["precision"],
        "b3_recall": b3["recall"],
        "b3_f1": b3["f1"],
        "predicted_cluster_count": len(groups),
        "true_cluster_count": len(truth_groups),
        "cluster_count_bias": len(groups) - len(truth_groups),
        "input_tokens": meta.get("input_tokens", 0),
        "output_tokens": meta.get("output_tokens", 0),
        "total_tokens": meta.get("total_tokens", 0),
        "latency": meta.get("latency", ""),
        "parse_success": True,
        "retry_count": meta.get("retry_count", 0),
        "raw_response_path": str(raw_path),
        "parsed_result_path": str(parsed_path),
        "error": "",
    }


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def avg(values):
    return sum(values) / len(values) if values else None


def med(values):
    return statistics.median(values) if values else None


def summarize(rows):
    ok = [r for r in rows if r["parse_success"]]
    summary = []
    for condition in CONDITIONS:
        part = [r for r in ok if r["condition"] == condition]
        summary.append({
            "condition": condition,
            "mean_b3_precision": avg([r["b3_precision"] for r in part]),
            "mean_b3_recall": avg([r["b3_recall"] for r in part]),
            "mean_b3_f1": avg([r["b3_f1"] for r in part]),
            "median_b3_f1": med([r["b3_f1"] for r in part]),
            "mean_pairwise_f1": avg([r["pairwise_f1"] for r in part]),
            "mean_predicted_cluster_count": avg([r["predicted_cluster_count"] for r in part]),
            "mean_cluster_count_bias": avg([r["cluster_count_bias"] for r in part]),
            "mean_input_tokens": avg([float(r["input_tokens"] or 0) for r in part]),
            "mean_output_tokens": avg([float(r["output_tokens"] or 0) for r in part]),
            "parse_failure_count": len([r for r in rows if r["condition"] == condition and not r["parse_success"]]),
        })
    baseline = {r["author_id"]: r for r in ok if r["condition"] == "minimal"}
    deltas = []
    for r in ok:
        if r["condition"] == "minimal" or r["author_id"] not in baseline:
            continue
        b = baseline[r["author_id"]]
        deltas.append({
            "author_id": r["author_id"],
            "difficulty_group": r["difficulty_group"],
            "condition": r["condition"],
            "minimal_b3_f1": b["b3_f1"],
            "condition_b3_f1": r["b3_f1"],
            "delta_b3_f1": r["b3_f1"] - b["b3_f1"],
            "minimal_pairwise_f1": b["pairwise_f1"],
            "condition_pairwise_f1": r["pairwise_f1"],
            "delta_pairwise_f1": r["pairwise_f1"] - b["pairwise_f1"],
            "minimal_cluster_count": b["predicted_cluster_count"],
            "condition_cluster_count": r["predicted_cluster_count"],
            "delta_cluster_count": r["predicted_cluster_count"] - b["predicted_cluster_count"],
        })
    return summary, deltas


def plot_html(title, body):
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script></head>
<body><div id="chart" style="width:100%;height:720px;"></div><script>{body}</script></body></html>"""


def write_plots(rows, deltas):
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    ok = [r for r in rows if r["parse_success"]]
    data = json.dumps(ok, ensure_ascii=False)
    PLOT_DIR.joinpath("condition_performance.html").write_text(plot_html("Condition Performance", f"""
const rows={data}; const conditions={json.dumps(list(CONDITIONS))};
const traces=conditions.map(c=>{{const r=rows.filter(x=>x.condition===c);return {{type:'box',name:c,y:r.map(x=>x.b3_f1),text:r.map(x=>`${{x.author_id}} (${{x.difficulty_group}})<br>B3=${{x.b3_f1.toFixed(4)}}<br>Pairwise=${{x.pairwise_f1.toFixed(4)}}`),boxpoints:'all',jitter:0.35}};}});
Plotly.newPlot('chart',traces,{{title:'B³ F1 by prompt condition',yaxis:{{title:'B³ F1'}}}});
"""), encoding="utf-8")
    delta_data = json.dumps(deltas, ensure_ascii=False)
    PLOT_DIR.joinpath("paired_prompt_comparison.html").write_text(plot_html("Paired Prompt Comparison", f"""
const rows={delta_data}; const conditions={json.dumps([c for c in CONDITIONS if c != 'minimal'])};
const traces=conditions.map(c=>{{const r=rows.filter(x=>x.condition===c);return {{type:'scatter',mode:'markers',name:c,x:r.map(x=>x.minimal_b3_f1),y:r.map(x=>x.condition_b3_f1),text:r.map(x=>`${{x.author_id}} delta=${{x.delta_b3_f1.toFixed(4)}}`)}};}});
traces.push({{type:'scatter',mode:'lines',name:'y=x',x:[0,1],y:[0,1],line:{{dash:'dash',color:'gray'}}}});
Plotly.newPlot('chart',traces,{{title:'Prompt condition vs minimal',xaxis:{{title:'minimal B³ F1'}},yaxis:{{title:'condition B³ F1'}}}});
"""), encoding="utf-8")
    authors = sorted({r["author_id"] for r in ok}, key=lambda a: ({"easy": 0, "medium": 1, "hard": 2}[next(r["difficulty_group"] for r in ok if r["author_id"] == a)], a))
    v = {(r["author_id"], r["condition"]): r for r in ok}
    z = [[v.get((a, c), {}).get("b3_f1") for c in CONDITIONS] for a in authors]
    PLOT_DIR.joinpath("author_prompt_heatmap.html").write_text(plot_html("Author Prompt Heatmap", f"""
Plotly.newPlot('chart',[{{type:'heatmap',x:{json.dumps(list(CONDITIONS))},y:{json.dumps(authors)},z:{json.dumps(z)},colorscale:'Viridis'}}],{{title:'Author-level B³ F1 by prompt'}});
"""), encoding="utf-8")


def format_value(value):
    if value is None:
        return ""
    return f"{value:.4f}" if isinstance(value, float) else str(value)


def md_table(rows, cols):
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(format_value(row.get(c, "")) for c in cols) + " |")
    return "\n".join(lines)


def metric_delta(by_condition, left, right, key):
    left_value = by_condition[left].get(key)
    right_value = by_condition[right].get(key)
    if left_value is None or right_value is None:
        return "NA"
    return f"{left_value - right_value:.4f}"


def write_report(selected, summary, deltas, model, rows=None):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rows = rows or []
    successful_summary = [row for row in summary if row["mean_b3_f1"] is not None]
    best = max(successful_summary, key=lambda r: r["mean_b3_f1"]) if successful_summary else None
    by_condition = {r["condition"]: r for r in summary}
    cases = sorted(deltas, key=lambda r: abs(r["delta_b3_f1"]), reverse=True)[:3]
    counts = Counter(x["difficulty_group"] for x in selected)
    success_count = sum(1 for row in rows if row.get("parse_success"))
    failure_count = sum(1 for row in rows if not row.get("parse_success"))
    failure_reasons = Counter(row.get("error", "") or "unknown" for row in rows if not row.get("parse_success"))
    failure_summary = "; ".join(f"{reason}: {count}" for reason, count in failure_reasons.items()) or "无"
    report = f"""# Prompt/context instruction 实验报告

## 1. 实验目的

固定作者、论文顺序和 T+O+C 数据，只改变 prompt 和 context instruction，观察任务说明、证据指导、错误提醒和分阶段分析的影响。

## 2. 实验设置

- 数据集：`{DATA_DIR}`
- 作者数量：{len(selected)}，easy={counts.get('easy', 0)}，medium={counts.get('medium', 0)}，hard={counts.get('hard', 0)}
- 模型：`{model}`，DeepSeek API，temperature=0
- 条件：`minimal`、`task_definition`、`evidence_guidance`、`error_aware`、`analyze_then_cluster`

## 3. Prompt 条件

`minimal` 只说明聚类任务；`task_definition` 明确姓名消歧背景；`evidence_guidance` 解释 T/O/C 证据局限；`error_aware` 提醒过拆分和过合并；`analyze_then_cluster` 要求先检查证据再输出最终 JSON。

## 4. 主要结果

{md_table(summary, ['condition', 'mean_b3_precision', 'mean_b3_recall', 'mean_b3_f1', 'median_b3_f1', 'mean_pairwise_f1', 'mean_cluster_count_bias', 'mean_input_tokens', 'mean_output_tokens', 'parse_failure_count'])}

## 5. 主要发现

- 平均 B³ F1 最高的是 `{best['condition'] if best else 'NA'}`，mean B³ F1={format_value(best['mean_b3_f1']) if best else 'NA'}；没有成功解析的条件不参与 best condition 判断。
- `task_definition` 相比 `minimal` 的 mean B³ F1 变化为 {metric_delta(by_condition, 'task_definition', 'minimal', 'mean_b3_f1')}。
- `evidence_guidance` 相比 `task_definition` 的 mean B³ F1 变化为 {metric_delta(by_condition, 'evidence_guidance', 'task_definition', 'mean_b3_f1')}。
- `error_aware` 相比 `task_definition` 的 mean cluster count bias 变化为 {metric_delta(by_condition, 'error_aware', 'task_definition', 'mean_cluster_count_bias')}。
- `analyze_then_cluster` 相比 `evidence_guidance` 的 mean output tokens 变化为 {metric_delta(by_condition, 'analyze_then_cluster', 'evidence_guidance', 'mean_output_tokens')}。

## 6. 典型案例

{md_table(cases, ['author_id', 'difficulty_group', 'condition', 'minimal_b3_f1', 'condition_b3_f1', 'delta_b3_f1', 'delta_cluster_count'])}

## 7. 结论

这是 3 作者小规模实验。本轮 {success_count} 个 author-condition 通过解析和 full-coverage 校验，{failure_count} 个失败；失败原因包括：{failure_summary}。在成功样本上，`{best['condition'] if best else 'NA'}` 的平均 B³ F1 最高，但该均值需要结合 parse failure count 解读：部分条件只在少数作者上成功，不能直接视为总体最优。当前结果说明 prompt 内容会改变聚类倾向，但还不能证明更详细 prompt 稳定更好；后续需要更稳的请求策略或分批处理来降低输出完整性风险。
"""
    (REPORT_DIR / "prompt_context_experiment_report_zh.md").write_text(report, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    raw = load_json(args.data_dir / "sna_valid_raw.json")
    papers = load_json(args.data_dir / "sna_valid_pub.json")
    selected = select_authors()
    rows = []
    for item in selected:
        name = item["author_id"]
        paper_ids = raw[name]
        cards = make_cards(name, paper_ids, papers)
        truth_groups = load_truth_groups(name, paper_ids)
        prompts = {condition: build_prompt(cards, condition) for condition in CONDITIONS}
        check_prompt_invariance(cards, prompts)
        for condition, prompt in prompts.items():
            write_text(RESULT_DIR / "prompts" / name / f"{condition}.txt", prompt)
        print(f"\n=== {name} ({item['difficulty_group']}, papers={len(paper_ids)}) ===", flush=True)
        if args.check_only:
            print(f"prompt_check: ok, prompt_chars={len(prompts['minimal'])}", flush=True)
            continue
        for condition in CONDITIONS:
            try:
                row = run_call(item, condition, cards, paper_ids, truth_groups, args.model, args.timeout, args.force)
                rows.append(row)
                print(f"{condition}: B3 F1={row['b3_f1']:.4f}", flush=True)
            except (ValueError, json.JSONDecodeError, KeyError, urllib.error.URLError, http.client.RemoteDisconnected, http.client.IncompleteRead, TimeoutError) as exc:
                print(f"{condition}: failed: {exc}", flush=True)
                rows.append({
                    "author_id": name, "difficulty_group": item["difficulty_group"], "condition": condition,
                    "paper_count": len(paper_ids), "pairwise_precision": "", "pairwise_recall": "", "pairwise_f1": "",
                    "b3_precision": "", "b3_recall": "", "b3_f1": "", "predicted_cluster_count": "",
                    "true_cluster_count": len(truth_groups), "cluster_count_bias": "",
                    "input_tokens": "", "output_tokens": "", "total_tokens": "", "latency": "",
                    "parse_success": False, "retry_count": 0, "raw_response_path": "", "parsed_result_path": "",
                    "error": str(exc),
                })

    if args.check_only:
        print("\n=== check-only summary ===")
        print(f"data_dir={args.data_dir}")
        print("selected_authors=" + ", ".join(x["author_id"] for x in selected))
        print(f"prompts={RESULT_DIR / 'prompts'}")
        return

    fields = ["author_id", "difficulty_group", "condition", "paper_count", "pairwise_precision", "pairwise_recall", "pairwise_f1", "b3_precision", "b3_recall", "b3_f1", "predicted_cluster_count", "true_cluster_count", "cluster_count_bias", "input_tokens", "output_tokens", "total_tokens", "latency", "parse_success", "retry_count", "raw_response_path", "parsed_result_path", "error"]
    write_csv(RESULT_DIR / "author_condition_metrics.csv", rows, fields)
    summary, deltas = summarize(rows)
    write_csv(RESULT_DIR / "condition_summary.csv", summary, list(summary[0]))
    write_csv(RESULT_DIR / "author_deltas.csv", deltas, list(deltas[0]) if deltas else ["author_id"])
    write_plots(rows, deltas)
    write_report(selected, summary, deltas, args.model, rows)

    print("\n=== summary ===")
    print(f"data_dir={args.data_dir}")
    print("selected_authors=" + ", ".join(x["author_id"] for x in selected))
    print(f"successful_calls={sum(1 for r in rows if r['parse_success'])}")
    print(f"failed_calls={sum(1 for r in rows if not r['parse_success'])}")
    for row in summary:
        print(f"{row['condition']}: mean B3 F1={format_value(row['mean_b3_f1'])}")
    successful_summary = [row for row in summary if row["mean_b3_f1"] is not None]
    print(f"best_condition={max(successful_summary, key=lambda r: r['mean_b3_f1'])['condition'] if successful_summary else 'NA'}")
    print(f"report={REPORT_DIR / 'prompt_context_experiment_report_zh.md'}")
    print(f"plots={PLOT_DIR}")
    print(f"results={RESULT_DIR}")


if __name__ == "__main__":
    main()
