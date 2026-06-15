#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common.llm_cluster import DEFAULT_DATA, FIELDS, paper_card, prompt_for


CONTEXT_LIMITS = {
    "deepseek-chat": 32000,
    "deepseek-reasoner": 32000,
    "minimax": 192000,
    "minimax-m2.7": 192000,
    "glm": 128000,
    "glm-5.1": 128000,
    "qwen": 256000,
    "qwen3.5-27b": 256000,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--name", action="append")
    parser.add_argument("--field", action="append", choices=FIELDS)
    parser.add_argument("--provider", choices=PROVIDERS, default="zhiyuan")
    parser.add_argument("--model")
    parser.add_argument("--max-prompt-tokens", type=int)
    parser.add_argument("--delay", type=int, default=65)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not args.model:
        args.model = "deepseek-v4-flash" if args.provider == "deepseek" else "deepseek-chat"

    raw = json.loads((args.data_dir / "sna_valid_raw.json").read_text())
    papers = json.loads((args.data_dir / "sna_valid_pub.json").read_text())
    names = args.name or list(raw)
    fields = args.field or list(FIELDS)
    dataset = args.data_dir.parts[-3] if len(args.data_dir.parts) >= 3 else args.data_dir.name
    context_safe_limit = (
        986000 if args.provider == "deepseek"
        else CONTEXT_LIMITS[args.model] - 14000
    )
    provider_limit = context_safe_limit if args.provider == "deepseek" else min(
        context_safe_limit, 80000
    )
    max_prompt_tokens = args.max_prompt_tokens or provider_limit
    print(
        f"provider={args.provider}, model={args.model}, prompt limit={max_prompt_tokens}, "
        f"delay={args.delay}s"
    )

    for name in names:
        for field in fields:
            cards = [paper_card(papers[pid], name, [field]) for pid in raw[name]]
            prompt_cards = cards
            if args.provider == "deepseek":
                prompt_cards = [
                    {"record_index": index, field: card[field]}
                    for index, card in enumerate(cards)
                ]
            estimated_tokens = len(prompt_for(
                name,
                [field],
                prompt_cards,
                compact_output=args.provider == "deepseek",
            )) // 4
            print(f"\n{name} / {field}: approximately {estimated_tokens} prompt tokens")
            result_path = Path("result/stage1_single_field") / dataset / f"{name}_{field}.json"
            if result_path.exists() and not args.force:
                print(f"skip: result already exists at {result_path}")
                continue
            if estimated_tokens > max_prompt_tokens:
                print(
                    f"skip: prompt exceeds the safe {max_prompt_tokens}-token limit "
                    "for this model/rate limit"
                )
                continue

            command = [
                sys.executable,
                "src/common/llm_cluster.py",
                "--data-dir", str(args.data_dir),
                "--name", name,
                "--field", field,
                "--provider", args.provider,
                "--model", args.model,
            ]
            result = subprocess.run(command, check=False)
            if result.returncode:
                print("the raw result may still be saved; continuing to the next experiment")

            subprocess.run([
                sys.executable,
                "src/stage1_single_field/report.py",
                "--data-dir", str(args.data_dir),
            ], check=True)
            time.sleep(args.delay)


if __name__ == "__main__":
    main()
