#!/usr/bin/env python3
import argparse
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common.llm_cluster import DEFAULT_DATA, FIELDS, truth_path
from src.common.metrics import analyze_result


def average(values):
    return sum(values) / len(values) if values else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--size", type=int, default=2, choices=range(2, 7))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    raw = json.loads((args.data_dir / "sna_valid_raw.json").read_text())
    truth = json.loads(truth_path(args.data_dir).read_text())
    dataset = args.data_dir.parts[-3]
    result_root = Path("result/stage2_field_combinations") / dataset / f"{args.size}_fields"
    combination_reports = []

    for fields in itertools.combinations(FIELDS, args.size):
        combination = "+".join(fields)
        combination_dir = result_root / combination
        name_results = []
        for name, paper_ids in raw.items():
            result_path = combination_dir / f"{name}.json"
            response_paths = list(combination_dir.glob(f"{name}.*.response.json"))
            item = {"name": name, "status": "not_run", "result_file": str(result_path)}
            if result_path.exists():
                clusters = json.loads(result_path.read_text()).get("clusters", [])
                item["status"] = "completed"
                item["analysis"] = analyze_result(clusters, paper_ids, truth[name])
            elif response_paths:
                response_path = max(response_paths, key=lambda path: path.stat().st_mtime)
                response = json.loads(response_path.read_text())
                choice = (response.get("choices") or [{}])[0]
                item.update({
                    "status": "failed",
                    "response_file": str(response_path),
                    "finish_reason": choice.get("finish_reason"),
                    "usage": response.get("usage", {}),
                })
            name_results.append(item)

        completed = [item for item in name_results if item["status"] == "completed"]
        recalls = [
            item["analysis"]["pairwise_metrics_missing_as_singletons"]["recall"]
            for item in completed
        ]
        precisions = [
            item["analysis"]["pairwise_metrics_missing_as_singletons"]["precision"]
            for item in completed
        ]
        coverage_rates = [
            item["analysis"]["complete_author_coverage"]["rate"]
            for item in completed
        ]
        combination_reports.append({
            "combination": combination,
            "fields": list(fields),
            "completed_names": len(completed),
            "failed_names": sum(item["status"] == "failed" for item in name_results),
            "average_pairwise_recall": average(recalls),
            "average_complete_author_coverage": average(coverage_rates),
            "average_pairwise_precision": average(precisions),
            "names": name_results,
        })

    ranking = sorted(
        combination_reports,
        key=lambda item: (
            item["average_complete_author_coverage"] is not None,
            item["average_complete_author_coverage"] or -1,
            item["average_pairwise_recall"] or -1,
        ),
        reverse=True,
    )
    report = {
        "dataset": dataset,
        "objective": "maximize recall while keeping every real author's papers together",
        "ranking_priority": [
            "average_complete_author_coverage",
            "average_pairwise_recall",
            "average_pairwise_precision_as_secondary_context",
        ],
        "ranking": [
            {
                key: item[key]
                for key in (
                    "combination",
                    "completed_names",
                    "failed_names",
                    "average_complete_author_coverage",
                    "average_pairwise_recall",
                    "average_pairwise_precision",
                )
            }
            for item in ranking
        ],
        "combinations": combination_reports,
    }
    output = args.output or result_root / "report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
