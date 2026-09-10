#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


from name_disambiguation.core.llm_cluster import validate


def recover_assignments(response):
    content = response["choices"][0]["message"]["content"]
    marker = '"assignments"'
    marker_position = content.find(marker)
    if marker_position < 0:
        raise ValueError("response does not contain assignments")
    colon = content.find(":", marker_position + len(marker))
    assignments, _ = json.JSONDecoder().raw_decode(content[colon + 1:].lstrip())
    return assignments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--response", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    raw = json.loads((args.data_dir / "sna_valid_raw.json").read_text())
    paper_ids = raw[args.name]
    assignments = recover_assignments(json.loads(args.response.read_text()))
    expected_indexes = {str(index) for index in range(len(paper_ids))}
    if set(assignments) != expected_indexes:
        raise ValueError(
            f"incomplete assignments: expected {len(expected_indexes)}, "
            f"found {len(assignments)}"
        )

    grouped = {}
    for index, cluster_id in assignments.items():
        grouped.setdefault(str(cluster_id), []).append(paper_ids[int(index)])
    result = {
        "clusters": [
            {
                "label": f"cluster {cluster_id}",
                "paper_ids": ids,
            }
            for cluster_id, ids in grouped.items()
        ]
    }
    validate(result, paper_ids)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(
        f"recovered {len(assignments)} assignments into "
        f"{len(result['clusters'])} clusters"
    )
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
