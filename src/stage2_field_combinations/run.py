#!/usr/bin/env python3
import argparse
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common.llm_cluster import DEFAULT_DATA, FIELDS, paper_card, prompt_for


def selected_combinations(size, requested):
    if requested:
        combinations = [tuple(item.split("+")) for item in requested]
        for fields in combinations:
            if len(fields) < 2 or any(field not in FIELDS for field in fields):
                raise ValueError(f"invalid combination: {'+'.join(fields)}")
        return combinations
    return list(itertools.combinations(FIELDS, size))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--name", action="append")
    parser.add_argument("--size", type=int, default=2, choices=range(2, 7))
    parser.add_argument("--combination", action="append")
    parser.add_argument("--provider", choices=("zhiyuan", "deepseek"), default="deepseek")
    parser.add_argument("--model")
    parser.add_argument("--delay", type=int, default=10)
    parser.add_argument("--max-prompt-tokens", type=int, default=900000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not args.model:
        args.model = "deepseek-v4-flash" if args.provider == "deepseek" else "deepseek-chat"

    raw = json.loads((args.data_dir / "sna_valid_raw.json").read_text())
    papers = json.loads((args.data_dir / "sna_valid_pub.json").read_text())
    names = args.name or list(raw)
    combinations = selected_combinations(args.size, args.combination)
    dataset = args.data_dir.parts[-3]
    result_root = Path("result/stage2_field_combinations") / dataset / f"{args.size}_fields"

    print(f"provider={args.provider}, model={args.model}, combinations={len(combinations)}")
    for fields in combinations:
        combination = "+".join(fields)
        print(f"\n=== {combination} ===")
        for name in names:
            output = result_root / combination / f"{name}.json"
            if output.exists() and not args.force:
                print(f"skip {name}: {output} already exists")
                continue

            cards = [paper_card(papers[pid], name, fields) for pid in raw[name]]
            prompt_cards = cards
            if args.provider == "deepseek":
                prompt_cards = [
                    {
                        "record_index": index,
                        **{field: card[field] for field in fields},
                    }
                    for index, card in enumerate(cards)
                ]
            prompt_tokens = len(prompt_for(
                name, fields, prompt_cards, compact_output=args.provider == "deepseek"
            )) // 4
            print(f"{name}: approximately {prompt_tokens} prompt tokens")
            if prompt_tokens > args.max_prompt_tokens:
                print("skip: prompt exceeds configured limit")
                continue

            command = [
                sys.executable,
                "src/common/llm_cluster.py",
                "--data-dir", str(args.data_dir),
                "--name", name,
                "--provider", args.provider,
                "--model", args.model,
                "--output", str(output),
            ]
            for field in fields:
                command.extend(["--field", field])
            result = subprocess.run(command, check=False)
            if result.returncode:
                print("failed; diagnostic response may have been saved")

            subprocess.run([
                sys.executable,
                "src/stage2_field_combinations/report.py",
                "--data-dir", str(args.data_dir),
                "--size", str(args.size),
            ], check=True)
            time.sleep(args.delay)


if __name__ == "__main__":
    main()
