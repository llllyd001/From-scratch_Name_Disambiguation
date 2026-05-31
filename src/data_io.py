import json
from pathlib import Path


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dump_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def raw_candidate_sets(raw):
    for target_key, paper_ids in raw.items():
        yield target_key, paper_ids


def ground_truth_index(ground_truth):
    info = {}
    index = 0

    for target_key, author_groups in ground_truth.items():
        for real_author_key, paper_ids in author_groups.items():
            for paper_id in paper_ids:
                info[paper_id] = {
                    "true_author_index": index,
                    "true_author_key": real_author_key,
                    "target_name": target_key,
                }
            index += 1

    return info
