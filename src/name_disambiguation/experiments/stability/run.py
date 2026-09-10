#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


from name_disambiguation.core.llm_cluster import DEFAULT_DATA, FIELDS
from name_disambiguation.paths import RESULTS_ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--combination", default="title+coauthors+organization")
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--start-run", type=int, default=1)
    parser.add_argument("--name", action="append")
    parser.add_argument("--provider", choices=("zhiyuan", "deepseek"), default="deepseek")
    parser.add_argument("--model")
    parser.add_argument("--delay", type=int, default=10)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    fields = args.combination.split("+")
    if len(fields) < 2 or any(field not in FIELDS for field in fields):
        parser.error(f"invalid combination: {args.combination}")
    if args.runs < 1:
        parser.error("--runs must be at least 1")
    if not args.model:
        args.model = "deepseek-v4-flash" if args.provider == "deepseek" else "deepseek-chat"

    raw = json.loads((args.data_dir / "sna_valid_raw.json").read_text())
    names = args.name or list(raw)
    unknown_names = set(names) - set(raw)
    if unknown_names:
        parser.error(f"unknown names: {', '.join(sorted(unknown_names))}")

    stability_dir = RESULTS_ROOT / "stability" / args.combination
    for run_number in range(args.start_run, args.start_run + args.runs):
        run_dir = stability_dir / f"run_{run_number}"
        print(f"\n=== run_{run_number} ===")
        for name in names:
            output = run_dir / f"{name}.json"
            if output.exists() and not args.force:
                print(f"skip {name}: {output} already exists")
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
            for field in fields:
                command.extend(["--field", field])
            result = subprocess.run(command, check=False)
            if result.returncode:
                print(f"failed: run_{run_number}/{name}")
            time.sleep(args.delay)

        subprocess.run([
            sys.executable,
            "-m", "name_disambiguation.experiments.stability.report",
            "--data-dir", str(args.data_dir),
            "--combination", args.combination,
        ], check=True)


if __name__ == "__main__":
    main()
