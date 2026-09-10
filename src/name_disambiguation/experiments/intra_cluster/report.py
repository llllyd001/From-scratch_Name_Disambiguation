#!/usr/bin/env python3
import argparse
import json
import statistics
from pathlib import Path


from name_disambiguation.core.llm_cluster import DEFAULT_DATA, FIELDS, truth_path
from name_disambiguation.core.metrics import analyze_result, pairwise_metrics
from name_disambiguation.paths import RESULTS_ROOT


DEFAULT_STAGE1_DIR = (
    RESULTS_ROOT / "field_selection" / "combinations" / "NA_Demo" / "3_fields"
    / "title+coauthors+organization"
)


def truth_subset(truth, paper_ids):
    paper_set = set(paper_ids)
    return {
        author_id: [pid for pid in ids if pid in paper_set]
        for author_id, ids in truth.items()
        if paper_set & set(ids)
    }


def composition(clusters, truth, paper_ids):
    return analyze_result(clusters, paper_ids, truth_subset(truth, paper_ids))[
        "cluster_compositions"
    ]


def compact_composition(clusters, truth, paper_ids):
    rows = []
    for item in composition(clusters, truth, paper_ids):
        rows.append({
            "cluster_index": item["cluster_index"],
            "llm_label": item["llm_label"],
            "paper_count": sum(item["composition"].values()),
            "composition": item["composition"],
        })
    return rows


def average(items, key):
    values = [item[key] for item in items]
    return statistics.mean(values) if values else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--stage1-dir", type=Path, default=DEFAULT_STAGE1_DIR)
    parser.add_argument("--stage2-root", type=Path)
    parser.add_argument("--field", action="append", choices=FIELDS)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    args.field = args.field or ["title", "abstract", "coauthors", "organization"]

    raw = json.loads((args.data_dir / "sna_valid_raw.json").read_text())
    truth = json.loads(truth_path(args.data_dir).read_text())
    dataset = args.data_dir.parts[-3]
    stage1_key = args.stage1_dir.name
    field_key = "+".join(args.field)
    stage2_root = args.stage2_root or (
        RESULTS_ROOT / "intra_cluster" / dataset / stage1_key / field_key
    )

    merged_prediction = {}
    names = []
    readable_summary = []
    for name, paper_ids in raw.items():
        stage1_path = args.stage1_dir / f"{name}.json"
        stage1_clusters = json.loads(stage1_path.read_text()).get("clusters", [])
        merged_clusters = []
        parent_reports = []
        for index, parent in enumerate(stage1_clusters):
            parent_ids = list(dict.fromkeys(parent.get("paper_ids", [])))
            stage2_path = stage2_root / name / f"cluster_{index}.json"
            if stage2_path.exists():
                stage2_result = json.loads(stage2_path.read_text())
                child_clusters = stage2_result.get("clusters", [])
                merged_clusters.extend(cluster["paper_ids"] for cluster in child_clusters)
                local_truth = truth_subset(truth[name], parent_ids)
                local_truth_groups = [set(ids) for ids in local_truth.values()]
                child_groups = [set(cluster["paper_ids"]) for cluster in child_clusters]
                input_composition = compact_composition([parent], truth[name], parent_ids)[0]
                input_composition["cluster_index"] = index
                output_composition = compact_composition(
                    child_clusters,
                    truth[name],
                    parent_ids,
                )
                readable_summary.append({
                    "name": name,
                    "stage1_cluster_index": index,
                    "input_cluster": input_composition,
                    "output_clusters": output_composition,
                    "stage2_file": str(stage2_path),
                })
                parent_reports.append({
                    "cluster_index": index,
                    "stage1_label": parent.get("label", ""),
                    "stage2_file": str(stage2_path),
                    "paper_count": len(parent_ids),
                    "input_cluster": input_composition,
                    "output_clusters": output_composition,
                    "stage2_pairwise_metrics_inside_parent": pairwise_metrics(
                        local_truth_groups,
                        child_groups,
                        parent_ids,
                    ),
                })
            else:
                merged_clusters.append(parent_ids)
        merged_prediction[name] = merged_clusters
        analysis = analyze_result(
            [{"paper_ids": ids} for ids in merged_clusters],
            paper_ids,
            truth[name],
        )
        names.append({
            "name": name,
            "stage1_file": str(stage1_path),
            "stage2_clusters_refined": len(parent_reports),
            "analysis_after_stage2": analysis,
            "refined_parent_clusters": parent_reports,
        })

    metrics = [
        item["analysis_after_stage2"]["pairwise_metrics_missing_as_singletons"]
        for item in names
    ]
    covered = sum(
        item["analysis_after_stage2"]["complete_author_coverage"]["fully_covered_authors"]
        for item in names
    )
    total_authors = sum(
        item["analysis_after_stage2"]["complete_author_coverage"]["total_authors"]
        for item in names
    )
    report = {
        "dataset": dataset,
        "stage1_dir": str(args.stage1_dir),
        "stage2_root": str(stage2_root),
        "fields": args.field,
        "stage2_refinement_summary": readable_summary,
        "summary": {
            "average_pairwise_precision": average(metrics, "precision"),
            "average_pairwise_recall": average(metrics, "recall"),
            "average_pairwise_f1": average(metrics, "f1"),
            "fully_covered_authors": covered,
            "total_authors": total_authors,
            "complete_author_coverage": covered / total_authors if total_authors else None,
        },
        "names": names,
    }
    output = args.output or stage2_root / "report.json"
    merged_output = output.with_name("merged_result.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    merged_output.write_text(json.dumps(merged_prediction, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"saved: {output}")
    print(f"saved: {merged_output}")


if __name__ == "__main__":
    main()
