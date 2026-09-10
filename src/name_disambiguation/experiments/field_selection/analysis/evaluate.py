#!/usr/bin/env python3
import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from name_disambiguation.paths import RESULTS_ROOT, dataset_dir


DATA = dataset_dir()


def normalize(name):
    return re.sub(r"[^a-z0-9]", "", name.lower())


def name_forms(name):
    words = re.findall(r"[a-z0-9]+", name.lower())
    return {"".join(words), "".join(reversed(words))}


def paper_info(paper, name):
    forms = name_forms(name)
    organizations = [
        author.get("org", "")
        for author in paper.get("authors", [])
        if normalize(author.get("name", "")) in forms
    ]
    return {
        "id": paper["id"],
        "title": paper.get("title", ""),
        "organizations": organizations,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="haifeng_qian")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    raw = json.loads((DATA / "sna_valid_raw.json").read_text())
    truth = json.loads((DATA / "sna_valid_ground_truth.json").read_text())
    papers = json.loads((DATA / "sna_valid_pub.json").read_text())
    if args.name not in raw:
        parser.error(f"name must be one of: {', '.join(raw)}")

    output_dir = RESULTS_ROOT / "field_selection" / "analysis"
    input_path = args.input or output_dir / f"{args.name}_llm_coarse.json"
    output_path = args.output or output_dir / f"{args.name}_llm_coarse_evaluated.json"
    clusters = json.loads(input_path.read_text())["clusters"]

    true_author_by_paper = {
        pid: author_id
        for author_id, ids in truth[args.name].items()
        for pid in ids
    }
    memberships = defaultdict(list)
    organized_clusters = []

    for index, cluster in enumerate(clusters):
        ids = cluster.get("paper_ids", [])
        counts = Counter(ids)
        grouped = defaultdict(list)
        unknown = []

        for pid in dict.fromkeys(ids):
            memberships[pid].append(index)
            if pid not in papers or pid not in true_author_by_paper:
                unknown.append(pid)
                continue
            grouped[true_author_by_paper[pid]].append(paper_info(papers[pid], args.name))

        organized_clusters.append({
            "cluster_index": index,
            "llm_label": cluster.get("label", ""),
            "paper_occurrences": len(ids),
            "unique_valid_papers": sum(map(len, grouped.values())),
            "true_authors": {
                author_id: {
                    "paper_count": len(items),
                    "papers": items,
                }
                for author_id, items in grouped.items()
            },
            "unknown_paper_ids": unknown,
            "duplicate_ids_inside_cluster": sorted(
                pid for pid, count in counts.items() if count > 1
            ),
        })

    raw_ids = set(raw[args.name])
    returned_ids = set(memberships)
    missing_by_author = defaultdict(list)
    for pid in sorted(raw_ids - returned_ids):
        author_id = true_author_by_paper[pid]
        missing_by_author[author_id].append(paper_info(papers[pid], args.name))

    report = {
        "name": args.name,
        "field_explanations": {
            "cluster_composition": "Each true author ID and its number of unique papers in this LLM cluster.",
            "cross_cluster_duplicates": "Paper IDs assigned by the LLM to more than one cluster. The numbers are cluster indexes.",
            "duplicate_ids_inside_cluster": "Paper IDs repeated more than once inside the same LLM cluster.",
            "unknown_paper_ids": "Paper IDs returned by the LLM that are not valid input IDs for this name.",
            "missing_papers": "Valid input papers that the LLM did not return in any cluster.",
        },
        "cluster_compositions": [
            {
                "cluster_index": cluster["cluster_index"],
                "llm_label": cluster["llm_label"],
                "composition": {
                    author_id: details["paper_count"]
                    for author_id, details in cluster["true_authors"].items()
                },
            }
            for cluster in organized_clusters
        ],
        "summary": {
            "input_papers": len(raw_ids),
            "llm_clusters": len(clusters),
            "returned_valid_unique_papers": len(returned_ids & raw_ids),
            "missing_papers": len(raw_ids - returned_ids),
            "unknown_paper_ids": sorted(returned_ids - raw_ids),
            "cross_cluster_duplicates": {
                pid: indexes
                for pid, indexes in memberships.items()
                if len(indexes) > 1
            },
        },
        "clusters": organized_clusters,
        "missing_papers_by_true_author": {
            author_id: {
                "paper_count": len(items),
                "papers": items,
            }
            for author_id, items in missing_by_author.items()
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
