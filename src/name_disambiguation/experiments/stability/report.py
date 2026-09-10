#!/usr/bin/env python3
import argparse
import json
import statistics
from pathlib import Path


from name_disambiguation.core.llm_cluster import DEFAULT_DATA, truth_path
from name_disambiguation.core.metrics import analyze_result
from name_disambiguation.paths import RESULTS_ROOT


def summarize(values):
    if not values:
        return None
    return {
        "mean": statistics.mean(values),
        "standard_deviation": statistics.pstdev(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def evaluate_run(run_name, run_dir, names, raw, truth):
    per_name = []
    for name in names:
        path = run_dir / f"{name}.json"
        if not path.exists():
            per_name.append({"name": name, "status": "missing", "result_file": str(path)})
            continue
        clusters = json.loads(path.read_text()).get("clusters", [])
        analysis = analyze_result(clusters, raw[name], truth[name])
        per_name.append({
            "name": name,
            "status": "completed",
            "result_file": str(path),
            "analysis": analysis,
        })

    completed = [item for item in per_name if item["status"] == "completed"]
    metrics = [
        item["analysis"]["pairwise_metrics_missing_as_singletons"]
        for item in completed
    ]
    covered = sum(
        item["analysis"]["complete_author_coverage"]["fully_covered_authors"]
        for item in completed
    )
    total_authors = sum(
        item["analysis"]["complete_author_coverage"]["total_authors"]
        for item in completed
    )
    return {
        "run": run_name,
        "run_dir": str(run_dir),
        "completed_names": len(completed),
        "summary": {
            "average_pairwise_precision": statistics.mean(
                item["precision"] for item in metrics
            ) if metrics else None,
            "average_pairwise_recall": statistics.mean(
                item["recall"] for item in metrics
            ) if metrics else None,
            "average_pairwise_f1": statistics.mean(
                item["f1"] for item in metrics
            ) if metrics else None,
            "fully_covered_authors": covered,
            "total_authors": total_authors,
            "overall_complete_author_coverage": (
                covered / total_authors if total_authors else None
            ),
        },
        "names": per_name,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument(
        "--combination",
        default="title+coauthors+organization",
    )
    parser.add_argument("--stability-dir", type=Path)
    parser.add_argument("--baseline-dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    raw = json.loads((args.data_dir / "sna_valid_raw.json").read_text())
    truth = json.loads(truth_path(args.data_dir).read_text())
    names = list(raw)
    dataset = args.data_dir.parts[-3]
    stability_dir = args.stability_dir or RESULTS_ROOT / "stability" / args.combination
    baseline_dir = args.baseline_dir or (
        RESULTS_ROOT / "field_selection" / "combinations"
        / dataset
        / f"{len(args.combination.split('+'))}_fields"
        / args.combination
    )

    run_dirs = [("baseline", baseline_dir)]
    run_dirs.extend(
        (path.name, path)
        for path in sorted(stability_dir.glob("run_*"))
        if path.is_dir()
    )
    runs = [
        evaluate_run(run_name, run_dir, names, raw, truth)
        for run_name, run_dir in run_dirs
    ]
    completed_runs = [run for run in runs if run["completed_names"] == len(names)]
    per_name_stability = {}
    for name in names:
        completed_name_results = [
            item
            for run in completed_runs
            for item in run["names"]
            if item["name"] == name and item["status"] == "completed"
        ]
        metrics = [
            item["analysis"]["pairwise_metrics_missing_as_singletons"]
            for item in completed_name_results
        ]
        coverage = [
            item["analysis"]["complete_author_coverage"]["rate"]
            for item in completed_name_results
        ]
        clusters = [
            item["analysis"]["llm_clusters"] for item in completed_name_results
        ]
        per_name_stability[name] = {
            "run_count": len(completed_name_results),
            "pairwise_precision": summarize([item["precision"] for item in metrics]),
            "pairwise_recall": summarize([item["recall"] for item in metrics]),
            "pairwise_f1": summarize([item["f1"] for item in metrics]),
            "complete_author_coverage": summarize(coverage),
            "cluster_count": summarize(clusters),
        }

    report = {
        "dataset": dataset,
        "combination": args.combination,
        "run_count": len(completed_runs),
        "stability_summary": {
            "average_pairwise_precision": summarize([
                run["summary"]["average_pairwise_precision"] for run in completed_runs
            ]),
            "average_pairwise_recall": summarize([
                run["summary"]["average_pairwise_recall"] for run in completed_runs
            ]),
            "average_pairwise_f1": summarize([
                run["summary"]["average_pairwise_f1"] for run in completed_runs
            ]),
            "overall_complete_author_coverage": summarize([
                run["summary"]["overall_complete_author_coverage"]
                for run in completed_runs
            ]),
        },
        "per_name_stability": per_name_stability,
        "runs": runs,
    }
    output = args.output or stability_dir / "stability_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report["stability_summary"], ensure_ascii=False, indent=2))
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
