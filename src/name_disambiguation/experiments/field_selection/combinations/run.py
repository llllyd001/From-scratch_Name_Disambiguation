#!/usr/bin/env python3
import argparse
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path


from name_disambiguation.core.llm_cluster import DEFAULT_DATA, FIELDS, load_truth, paper_card, prompt_for
from name_disambiguation.experiments.field_selection.combinations.raw_result import write_raw_result
from name_disambiguation.paths import RESULTS_ROOT


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
    parser.add_argument("--ground-truth", type=Path)
    parser.add_argument("--raw-result-dir", type=Path)
    parser.add_argument("--delay", type=int, default=10)
    parser.add_argument("--max-prompt-tokens", type=int, default=900000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not args.model:
        args.model = "deepseek-v4-flash" if args.provider == "deepseek" else "deepseek-chat"

    raw = json.loads((args.data_dir / "sna_valid_raw.json").read_text())
    truth = load_truth(args.data_dir, args.ground_truth)
    papers = json.loads((args.data_dir / "sna_valid_pub.json").read_text())
    names = args.name or list(raw)
    combinations = selected_combinations(args.size, args.combination)
    dataset = args.data_dir.parts[-3]
    result_root = (
        RESULTS_ROOT / "field_selection" / "combinations" / dataset
        / f"{args.size}_fields"
    )

    print(f"provider={args.provider}, model={args.model}, combinations={len(combinations)}")
    for fields in combinations:
        combination = "+".join(fields)
        print(f"\n=== {combination} ===")
        for name in names:
            output = result_root / combination / f"{name}.json"
            raw_output = (
                args.raw_result_dir / f"{name}.json"
                if args.raw_result_dir else None
            )
            if raw_output and raw_output.exists() and not args.force:
                print(f"skip {name}: {raw_output} already exists")
                continue
            if output.exists() and not args.force:
                if raw_output:
                    saved = write_raw_result(
                        name,
                        output,
                        raw[name],
                        truth[name],
                        args.raw_result_dir,
                    )
                    print(f"saved: {saved}")
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
                "-m", "name_disambiguation.core.llm_cluster",
                "--data-dir", str(args.data_dir),
                "--name", name,
                "--provider", args.provider,
                "--model", args.model,
                "--output", str(output),
            ]
            if args.ground_truth:
                command.extend(["--ground-truth", str(args.ground_truth)])
            for field in fields:
                command.extend(["--field", field])
            result = subprocess.run(command, check=False)
            if result.returncode:
                print("failed; diagnostic response may have been saved")
            elif args.raw_result_dir:
                raw_output = write_raw_result(
                    name,
                    output,
                    raw[name],
                    truth[name],
                    args.raw_result_dir,
                )
                print(f"saved: {raw_output}")

            report_command = [
                sys.executable,
                "-m", "name_disambiguation.experiments.field_selection.combinations.report",
                "--data-dir", str(args.data_dir),
                "--size", str(args.size),
            ]
            if args.ground_truth:
                report_command.extend(["--ground-truth", str(args.ground_truth)])
            subprocess.run(report_command, check=True)
            time.sleep(args.delay)


if __name__ == "__main__":
    main()
