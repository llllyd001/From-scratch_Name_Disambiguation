#!/usr/bin/env python3
import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path


from openai import OpenAI
from name_disambiguation.core.llm_cluster import DEFAULT_DATA, FIELDS, paper_card, truth_path
from name_disambiguation.core.normalize_llm_response import normalize_response
from name_disambiguation.paths import RESULTS_ROOT


DEFAULT_STAGE1_DIR = (
    RESULTS_ROOT / "field_selection" / "combinations" / "NA_Demo" / "3_fields"
    / "title+coauthors+organization"
)


def provider_defaults(provider, model):
    if model:
        return model
    return "deepseek-v4-flash" if provider == "deepseek" else "deepseek-chat"


def prompt_for(name, fields, cards, parent_label):
    field_text = ", ".join(fields)
    return f"""You are doing the second stage of author name disambiguation.

These records all came from one broad stage-1 cluster for normalized name {name!r}.
The stage-1 label was: {parent_label!r}.

Your task is to split this parent cluster into real authors. It is valid to keep
all records in one cluster if the evidence points to one real person. Split only
when the supplied fields give clear evidence for different real authors. Do not
split papers that likely belong to the same real author.

Use only these supplied fields: {field_text}.
Return JSON using exactly this shape:
{{"assignments":{{"0":"0","1":"1"}},"labels":{{"0":"short profile"}}}}
Every input record_index must be a key in assignments exactly once.

PAPERS:
{json.dumps(cards, ensure_ascii=False, separators=(",", ":"))}
"""


def validate(result, paper_ids):
    known = set(paper_ids)
    assigned = []
    for cluster in result.get("clusters", []):
        ids = cluster.get("paper_ids", [])
        cluster["paper_ids"] = list(dict.fromkeys(ids))
        assigned.extend(cluster["paper_ids"])
    counts = Counter(assigned)
    duplicate = sorted(pid for pid, count in counts.items() if count > 1)
    unknown = sorted(set(assigned) - known)
    missing = sorted(known - set(assigned))
    if duplicate:
        raise ValueError(f"paper IDs assigned multiple times: {duplicate[:5]}")
    if unknown:
        raise ValueError(f"unknown paper IDs: {unknown[:5]}")
    if missing:
        raise ValueError(f"missing {len(missing)} paper IDs: {missing[:5]}")


def true_author(pid, truth):
    for author_id, paper_ids in truth.items():
        if pid in paper_ids:
            return author_id
    return None


def selected_indexes(stage1_clusters, args, truth):
    if args.cluster_index:
        return args.cluster_index
    if args.all_clusters:
        return [
            index for index, cluster in enumerate(stage1_clusters)
            if len(cluster.get("paper_ids", [])) > 1
        ]
    if args.mixed_only:
        indexes = []
        for index, cluster in enumerate(stage1_clusters):
            authors = {
                true_author(pid, truth)
                for pid in cluster.get("paper_ids", [])
            }
            authors.discard(None)
            if len(authors) > 1:
                indexes.append(index)
        return indexes
    raise ValueError("choose --mixed-only, --all-clusters, or --cluster-index")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--stage1-dir", type=Path, default=DEFAULT_STAGE1_DIR)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--name", action="append")
    parser.add_argument("--field", action="append", choices=FIELDS)
    parser.add_argument("--cluster-index", type=int, action="append")
    parser.add_argument("--mixed-only", action="store_true")
    parser.add_argument("--all-clusters", action="store_true")
    parser.add_argument("--provider", choices=("zhiyuan", "deepseek"), default="deepseek")
    parser.add_argument("--model")
    parser.add_argument("--delay", type=int, default=10)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.field = args.field or ["title", "abstract", "coauthors", "organization"]
    args.model = provider_defaults(args.provider, args.model)

    raw = json.loads((args.data_dir / "sna_valid_raw.json").read_text())
    papers = json.loads((args.data_dir / "sna_valid_pub.json").read_text())
    truth = json.loads(truth_path(args.data_dir).read_text())
    names = args.name or list(raw)
    dataset = args.data_dir.parts[-3]
    field_key = "+".join(args.field)
    stage1_key = args.stage1_dir.name
    output_root = args.output_root or (
        RESULTS_ROOT / "intra_cluster" / dataset / stage1_key / field_key
    )

    client = None
    if not args.dry_run:
        env_prefix = "DEEPSEEK" if args.provider == "deepseek" else "ZHIYUAN"
        client = OpenAI(
            api_key=os.environ[f"{env_prefix}_API_KEY"],
            base_url=os.environ[f"{env_prefix}_BASE_URL"],
            timeout=300,
            max_retries=0,
        )
    request_options = {}
    if args.provider == "deepseek":
        request_options = {
            "response_format": {"type": "json_object"},
            "extra_body": {"thinking": {"type": "disabled"}},
        }

    for name in names:
        stage1_path = args.stage1_dir / f"{name}.json"
        stage1_clusters = json.loads(stage1_path.read_text()).get("clusters", [])
        indexes = selected_indexes(stage1_clusters, args, truth[name])
        print(f"\n{name}: {len(indexes)} parent clusters selected")
        for index in indexes:
            parent = stage1_clusters[index]
            parent_ids = list(dict.fromkeys(parent.get("paper_ids", [])))
            if len(parent_ids) <= 1:
                print(f"skip cluster_{index}: singleton")
                continue
            output = output_root / name / f"cluster_{index}.json"
            if output.exists() and not args.force:
                print(f"skip cluster_{index}: {output} already exists")
                continue

            cards = [
                {
                    "record_index": record_index,
                    **{
                        field: paper_card(papers[pid], name, args.field)[field]
                        for field in args.field
                    },
                }
                for record_index, pid in enumerate(parent_ids)
            ]
            prompt = prompt_for(name, args.field, cards, parent.get("label", ""))
            print(f"cluster_{index}: {len(parent_ids)} papers, about {len(prompt) // 4} prompt tokens")
            if args.dry_run:
                continue
            response = client.chat.completions.create(
                model=args.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=60000 if args.provider == "deepseek" else 12000,
                **request_options,
            )

            output.parent.mkdir(parents=True, exist_ok=True)
            try:
                result = normalize_response(response.model_dump(), paper_ids=parent_ids)
                validate(result, parent_ids)
            except (ValueError, json.JSONDecodeError):
                response_path = output.with_suffix(f".{args.provider}.response.json")
                response_path.write_text(response.model_dump_json(indent=2) + "\n")
                print(f"saved failed response for diagnosis: {response_path}")
                raise

            result.update({
                "name": name,
                "stage1_cluster_index": index,
                "stage1_label": parent.get("label", ""),
                "fields": args.field,
                "input_paper_ids": parent_ids,
            })
            output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
            print(f"saved: {output}")
            time.sleep(args.delay)


if __name__ == "__main__":
    main()
