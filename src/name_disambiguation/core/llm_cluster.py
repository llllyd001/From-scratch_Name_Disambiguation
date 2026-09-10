#!/usr/bin/env python3
import argparse
import json
import os
import re
from pathlib import Path

from openai import OpenAI
from name_disambiguation.core.normalize_llm_response import normalize_response
from name_disambiguation.paths import RESULTS_ROOT, dataset_dir


DEFAULT_DATA = dataset_dir()
FIELDS = ("title", "abstract", "keywords", "coauthors", "organization", "venue", "year")
MODELS = ("deepseek-chat", "deepseek-reasoner", "minimax", "minimax-m2.7", "glm", "glm-5.1", "qwen", "qwen3.5-27b")
PROVIDERS = ("zhiyuan", "deepseek")


def truth_path(data_dir):
    ground_truth = data_dir / "sna_valid_ground_truth.json"
    return ground_truth if ground_truth.exists() else data_dir / "sna_valid_example.json"


def load_truth(data_dir, ground_truth=None):
    return json.loads((ground_truth or truth_path(data_dir)).read_text())


def normalize(name):
    return re.sub(r"[^a-z0-9]", "", name.lower())


# only lowercase letter here, both the normal and reversed order given
def name_forms(name):
    words = re.findall(r"[a-z0-9]+", name.lower())
    return {"".join(words), "".join(reversed(words))}


def field_value(paper, target_name, field):
    targets = name_forms(target_name)
    if field == "organization":
        return [
            author.get("org", "")
            for author in paper.get("authors", [])
            if normalize(author.get("name", "")) in targets
        ]
    if field == "coauthors":
        return [
            author.get("name", "")
            for author in paper.get("authors", [])
            if normalize(author.get("name", "")) not in targets
        ]
    return paper.get(field, "")


def paper_card(paper, target_name, fields):
    if isinstance(fields, str):
        fields = [fields]
    return {
        "id": paper["id"],
        **{field: field_value(paper, target_name, field) for field in fields},
    }


def prompt_for(name, fields, cards, compact_output=False):
    if isinstance(fields, str):
        fields = [fields]
    field_text = ", ".join(fields)
    output_format = (
        'Return JSON using exactly this shape:\n'
        '{"assignments":{"0":"0","1":"1"},'
        '"labels":{"0":"short profile","1":"short profile"}}\n'
        'Every input record_index must be a key in assignments exactly once.'
        if compact_output else
        'Return JSON using exactly this shape:\n'
        '{"clusters":[{"label":"short profile","paper_ids":["id1","id2"]}]}\n'
        'Every input paper ID must occur exactly once.'
    )
    return f"""You are performing high-recall author name disambiguation.

All records below contain an author whose normalized name is {name!r}. Different
real people may share this name. Create broad candidate clusters for a later,
finer disambiguation stage.

Titles provide topic evidence; organizations provide affiliation evidence;
coauthors provide relationship evidence. None of these fields is individually
decisive. This is not topic clustering: one real author may change topics,
institutions, collaborators, or publication styles over time.

The overriding rule is recall: papers by the same real person must not be split
across different clusters. Each paper must belong to exactly one cluster. When
evidence is uncertain, place it in the broader most plausible cluster. For this
experiment, infer identity using only these supplied fields: {field_text}.
Avoid oversplitting one real author only because topic, organization, or
collaborators change. Avoid overmerging different people only because one weak
signal overlaps. Do not create a catch-all cluster containing nearly every
paper.

First reason globally about stable author profiles and links across the complete
collection. Then return JSON only.
{output_format}
Use only supplied paper IDs.

PAPERS:
{json.dumps(cards, ensure_ascii=False, separators=(",", ":"))}
"""


def validate(result, paper_ids):
    clusters = result.get("clusters")
    if not isinstance(clusters, list) or not clusters:
        raise ValueError("response does not contain a non-empty clusters list")
    known = set(paper_ids)
    groups = []
    assigned = set()
    for cluster in clusters:
        ids = cluster.get("paper_ids", [])
        if not isinstance(ids, list):
            raise ValueError("paper_ids must be a list")
        ids = list(dict.fromkeys(ids))
        cluster["paper_ids"] = ids
        unknown = set(ids) - known
        if unknown:
            raise ValueError(f"unknown paper IDs: {sorted(unknown)[:5]}")
        repeated = set(ids) & assigned
        if repeated:
            raise ValueError(f"paper IDs assigned to multiple clusters: {sorted(repeated)[:5]}")
        assigned.update(ids)
        groups.append(set(ids))
    missing = known - assigned
    if missing:
        raise ValueError(f"missing {len(missing)} paper IDs: {sorted(missing)}")
    return groups


def evaluate(groups, truth):
    truth_iter = truth.values() if isinstance(truth, dict) else truth
    truth_groups = [set(ids) for ids in truth_iter]
    covered = [any(group >= real_author for group in groups) for real_author in truth_groups]
    total_papers = len(set().union(*truth_groups))
    return {
        "real_authors_fully_covered": sum(covered),
        "real_authors_total": len(covered),
        "coarse_recall": sum(covered) / len(covered),
        "coarse_clusters": len(groups),
        "largest_cluster": max(map(len, groups)),
        "largest_cluster_ratio": max(map(len, groups)) / total_papers,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--name", default="haifeng_qian")
    parser.add_argument("--field", action="append", choices=FIELDS)
    parser.add_argument("--provider", choices=PROVIDERS, default="zhiyuan")
    parser.add_argument("--model")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--ground-truth", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.field = args.field or ["organization"]
    if not args.model:
        args.model = "deepseek-v4-flash" if args.provider == "deepseek" else "deepseek-chat"
    if args.provider == "zhiyuan" and args.model not in MODELS:
        parser.error(f"unsupported Zhiyuan model: {args.model}")
    raw = json.loads((args.data_dir / "sna_valid_raw.json").read_text())
    truth_all = load_truth(args.data_dir, args.ground_truth)
    papers = json.loads((args.data_dir / "sna_valid_pub.json").read_text())
    if args.name not in raw:
        parser.error(f"name must be one of: {', '.join(raw)}")

    truth = truth_all[args.name]
    paper_ids = raw[args.name]
    cards = [paper_card(papers[pid], args.name, args.field) for pid in paper_ids]
    prompt_cards = cards
    if args.provider == "deepseek":
        prompt_cards = [
            {
                "record_index": index,
                **{field: card[field] for field in args.field},
            }
            for index, card in enumerate(cards)
        ]
    prompt = prompt_for(
        args.name,
        args.field,
        prompt_cards,
        compact_output=args.provider == "deepseek",
    )
    if args.dry_run:
        print(json.dumps({
            "name": args.name,
            "fields": args.field,
            "papers": len(cards),
            "ground_truth_authors": len(truth),
            "prompt_characters": len(prompt),
            "papers_with_nonempty_fields": {
                field: sum(bool(c[field]) for c in cards) for field in args.field
            },
        }, indent=2))
        return

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
    max_tokens = 60000 if args.provider == "deepseek" else 12000
    response = client.chat.completions.create(
        model=args.model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=max_tokens,
        **request_options,
    )

    dataset = args.data_dir.parts[-3] if len(args.data_dir.parts) >= 3 else args.data_dir.name
    field_key = "+".join(args.field)
    output = args.output or (
        RESULTS_ROOT / "field_selection" / "combinations" / dataset
        / f"{args.name}_{field_key}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    response_path = output.with_suffix(f".{args.provider}.response.json")
    try:
        result = normalize_response(
            response.model_dump(),
            paper_ids=paper_ids if args.provider == "deepseek" else None,
        )
        groups = validate(result, paper_ids)
    except (ValueError, json.JSONDecodeError):
        response_path.write_text(response.model_dump_json(indent=2) + "\n")
        print(f"saved failed response for diagnosis: {response_path}")
        raise
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    for failed_response in output.parent.glob(
        f"{output.stem}.*.response.json"
    ):
        failed_response.unlink()
    report = evaluate(groups, truth)
    print(json.dumps(report, indent=2))
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
