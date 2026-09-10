#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


from name_disambiguation.core.llm_cluster import DEFAULT_DATA, load_truth
from name_disambiguation.paths import RESULTS_ROOT


def write_raw_result(name, result_path, paper_ids, truth, output_dir):
    result = json.loads(result_path.read_text())
    truth_by_paper = {
        paper_id: author_id
        for author_id, ids in truth.items()
        for paper_id in ids
    }
    clusters = []
    for index, cluster in enumerate(result.get("clusters", [])):
        ids = cluster.get("paper_ids", [])
        clusters.append({
            "cluster_index": index,
            "label": cluster.get("label", ""),
            "paper_ids": ids,
            "papers": [
                {
                    "paper_id": paper_id,
                    "true_author_id": truth_by_paper[paper_id],
                }
                for paper_id in ids
            ],
        })

    returned = {
        paper_id
        for cluster in clusters
        for paper_id in cluster["paper_ids"]
    }
    expected = set(paper_ids)
    if returned != expected:
        raise ValueError(
            f"{name}: result paper IDs differ from raw data "
            f"(missing={len(expected - returned)}, unknown={len(returned - expected)})"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{name}.json"
    output.write_text(json.dumps({
        "name": name,
        "fields": ["title", "coauthors", "organization"],
        "source_result": str(result_path),
        "paper_count": len(paper_ids),
        "true_author_count": len(truth),
        "clusters": clusters,
    }, ensure_ascii=False, indent=2) + "\n")
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--ground-truth", type=Path)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=RESULTS_ROOT / "baseline_clusters"
    )
    parser.add_argument("--name", action="append")
    args = parser.parse_args()

    raw = json.loads((args.data_dir / "sna_valid_raw.json").read_text())
    truth = load_truth(args.data_dir, args.ground_truth)
    names = args.name or list(raw)
    for name in names:
        result_path = args.result_dir / f"{name}.json"
        if not result_path.exists():
            print(f"skip {name}: {result_path} does not exist")
            continue
        output = write_raw_result(
            name,
            result_path,
            raw[name],
            truth[name],
            args.output_dir,
        )
        print(f"saved: {output}")


if __name__ == "__main__":
    main()
