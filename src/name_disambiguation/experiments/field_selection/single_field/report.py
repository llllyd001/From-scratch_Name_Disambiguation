#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


from name_disambiguation.core.llm_cluster import (
    DEFAULT_DATA,
    FIELDS,
    field_value,
    paper_card,
    prompt_for,
    truth_path,
)
from name_disambiguation.core.metrics import analyze_result
from name_disambiguation.paths import RESULTS_ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--name", action="append")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    raw = json.loads((args.data_dir / "sna_valid_raw.json").read_text())
    truth = json.loads(truth_path(args.data_dir).read_text())
    papers = json.loads((args.data_dir / "sna_valid_pub.json").read_text())
    names = args.name or list(raw)
    unknown_names = set(names) - set(raw)
    if unknown_names:
        parser.error(f"unknown names: {', '.join(sorted(unknown_names))}")

    dataset = args.data_dir.parts[-3] if len(args.data_dir.parts) >= 3 else args.data_dir.name
    result_dir = RESULTS_ROOT / "field_selection" / "single_field" / dataset
    name_reports = []

    for name in names:
        paper_ids = raw[name]
        experiments = []
        for field in FIELDS:
            values = [field_value(papers[pid], name, field) for pid in paper_ids]
            cards = [paper_card(papers[pid], name, [field]) for pid in paper_ids]
            prompt = prompt_for(name, [field], cards)
            result_path = result_dir / f"{name}_{field}.json"
            metadata_path = result_path.with_suffix(".metadata.json")
            response_paths = list(result_dir.glob(f"{name}_{field}.*.response.json"))
            experiment = {
                "field": field,
                "papers": len(paper_ids),
                "papers_with_nonempty_field": sum(bool(value) for value in values),
                "papers_with_empty_field": sum(not bool(value) for value in values),
                "prompt_characters": len(prompt),
                "approximate_prompt_tokens": len(prompt) // 4,
                "single_request_feasible_with_deepseek_chat": len(prompt) // 4 <= 18000,
                "result_file": str(result_path),
                "status": "not_run",
            }
            if result_path.exists():
                clusters = json.loads(result_path.read_text()).get("clusters", [])
                experiment["status"] = "completed"
                if metadata_path.exists():
                    experiment["api_metadata"] = json.loads(metadata_path.read_text())
                experiment["result_analysis"] = analyze_result(
                    clusters, paper_ids, truth[name]
                )
            elif response_paths:
                response_path = max(response_paths, key=lambda path: path.stat().st_mtime)
                response = json.loads(response_path.read_text())
                choice = (response.get("choices") or [{}])[0]
                experiment["status"] = "failed"
                experiment["failed_response_file"] = str(response_path)
                experiment["failure"] = {
                    "finish_reason": choice.get("finish_reason"),
                    "usage": response.get("usage", {}),
                    "reason": (
                        "output reached the token limit before completing valid JSON"
                        if choice.get("finish_reason") == "length"
                        else "API response could not be normalized"
                    ),
                }
            experiments.append(experiment)
        name_reports.append({
            "name": name,
            "status_summary": {
                "completed": sum(item["status"] == "completed" for item in experiments),
                "failed": sum(item["status"] == "failed" for item in experiments),
                "not_run": sum(item["status"] == "not_run" for item in experiments),
            },
            "experiments": experiments,
        })

    field_summary = []
    for field in FIELDS:
        completed = [
            experiment
            for name_report in name_reports
            for experiment in name_report["experiments"]
            if experiment["field"] == field and experiment["status"] == "completed"
        ]
        metrics = [
            item["result_analysis"]["pairwise_metrics_missing_as_singletons"]
            for item in completed
        ]
        coverage_rates = [
            item["result_analysis"]["complete_author_coverage"]["rate"]
            for item in completed
        ]
        field_summary.append({
            "field": field,
            "completed_names": len(completed),
            "average_complete_author_coverage": (
                sum(coverage_rates) / len(coverage_rates) if coverage_rates else None
            ),
            "average_pairwise_precision": (
                sum(item["precision"] for item in metrics) / len(metrics) if metrics else None
            ),
            "average_pairwise_recall": (
                sum(item["recall"] for item in metrics) / len(metrics) if metrics else None
            ),
            "average_pairwise_f1": (
                sum(item["f1"] for item in metrics) / len(metrics) if metrics else None
            ),
        })

    report = {
        "dataset": dataset,
        "data_dir": str(args.data_dir),
        "name_summary": {
            item["name"]: item["status_summary"] for item in name_reports
        },
        "field_summary": field_summary,
        "names": name_reports,
    }
    output = args.output or result_dir / "single_field_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
