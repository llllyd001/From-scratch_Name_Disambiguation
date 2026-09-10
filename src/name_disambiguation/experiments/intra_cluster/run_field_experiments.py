#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
from pathlib import Path


from name_disambiguation.core.llm_cluster import DEFAULT_DATA, FIELDS
from name_disambiguation.experiments.intra_cluster.report import DEFAULT_STAGE1_DIR
from name_disambiguation.paths import RESULTS_ROOT


DEFAULT_COMBINATIONS = [
    "abstract+coauthors+venue",
    "abstract+organization+venue",
    "keywords+coauthors+organization",
    "keywords+coauthors+venue",
    "title+abstract+coauthors+venue",
    "title+abstract+organization+venue",
    "title+keywords+coauthors+organization",
    "abstract+keywords+coauthors+organization",
    "abstract+coauthors+organization+venue",
    "title+coauthors+organization+venue",
]


def parse_combination(text):
    fields = text.split("+")
    if len(fields) < 2 or any(field not in FIELDS for field in fields):
        raise ValueError(f"invalid combination: {text}")
    if len(set(fields)) != len(fields):
        raise ValueError(f"duplicate field in combination: {text}")
    return fields


def result_root(data_dir, stage1_dir):
    dataset = data_dir.parts[-3]
    return RESULTS_ROOT / "intra_cluster" / dataset / stage1_dir.name


def summarize(data_dir, stage1_dir, combinations, output):
    root = result_root(data_dir, stage1_dir)
    existing = sorted(
        path.parent.name
        for path in root.glob("*/report.json")
        if path.parent.is_dir()
    )
    combinations = sorted(set(combinations) | set(existing))
    reports = []
    for combination in combinations:
        report_path = root / combination / "report.json"
        if not report_path.exists():
            reports.append({
                "combination": combination,
                "status": "not_run",
                "report_file": str(report_path),
            })
            continue
        report = json.loads(report_path.read_text())
        summary = report.get("summary", {})
        refined = report.get("stage2_refinement_summary", [])
        reports.append({
            "combination": combination,
            "status": "completed",
            "report_file": str(report_path),
            "merged_result_file": str(report_path.with_name("merged_result.json")),
            "refined_parent_clusters": len(refined),
            **summary,
        })

    ranking = sorted(
        reports,
        key=lambda item: (
            item.get("status") == "completed",
            item.get("average_pairwise_f1") or -1,
            item.get("average_pairwise_precision") or -1,
        ),
        reverse=True,
    )
    summary_report = {
        "stage1_dir": str(stage1_dir),
        "objective": "compare stage2 intra-cluster field combinations",
        "combinations": reports,
        "ranking_by_average_pairwise_f1": ranking,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary_report, ensure_ascii=False, indent=2) + "\n")
    print(f"saved: {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--stage1-dir", type=Path, default=DEFAULT_STAGE1_DIR)
    parser.add_argument("--combination", action="append")
    parser.add_argument("--provider", choices=("zhiyuan", "deepseek"), default="deepseek")
    parser.add_argument("--model")
    parser.add_argument("--delay", type=int, default=10)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    combinations = args.combination or DEFAULT_COMBINATIONS
    parsed = [parse_combination(item) for item in combinations]
    root = result_root(args.data_dir, args.stage1_dir)
    summary_output = args.output or root / "stage2_field_experiment_report.json"

    if not args.summary_only:
        print(f"stage2 combinations: {len(parsed)}")
        for fields in parsed:
            combination = "+".join(fields)
            print(f"\n=== {combination} ===")
            run_command = [
                sys.executable,
                "-m", "name_disambiguation.experiments.intra_cluster.run",
                "--data-dir", str(args.data_dir),
                "--stage1-dir", str(args.stage1_dir),
                "--mixed-only",
                "--provider", args.provider,
                "--delay", str(args.delay),
            ]
            if args.model:
                run_command.extend(["--model", args.model])
            if args.force:
                run_command.append("--force")
            if args.dry_run:
                run_command.append("--dry-run")
            for field in fields:
                run_command.extend(["--field", field])
            result = subprocess.run(run_command, check=False)
            if result.returncode:
                print(f"failed: {combination}")
                continue
            if args.dry_run:
                continue

            report_command = [
                sys.executable,
                "-m", "name_disambiguation.experiments.intra_cluster.report",
                "--data-dir", str(args.data_dir),
                "--stage1-dir", str(args.stage1_dir),
            ]
            for field in fields:
                report_command.extend(["--field", field])
            subprocess.run(report_command, check=True)

    if not args.dry_run:
        summarize(args.data_dir, args.stage1_dir, ["+".join(fields) for fields in parsed], summary_output)


if __name__ == "__main__":
    main()
