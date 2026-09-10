#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path


def parse_json_content(content):
    content = (content or "").strip()
    content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content)
    if not content:
        raise ValueError("the model returned empty final content")
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start >= 0 and end > start:
            return json.loads(content[start:end + 1])
        raise


def normalize_response(response, paper_ids=None):
    if "error" in response:
        raise ValueError(f"API error: {response['error']}")
    choices = response.get("choices", [])
    if not choices:
        raise ValueError("API response contains no choices")

    choice = choices[0]
    message = choice.get("message", {})
    content = message.get("content", "")
    if not content.strip():
        reasoning = message.get("reasoning_content", "")
        if choice.get("finish_reason") == "length" and reasoning:
            raise ValueError(
                "thinking consumed the output budget and was truncated before "
                "the final JSON; rerun with thinking disabled"
            )
        raise ValueError("the model returned empty final content")

    if choice.get("finish_reason") == "length":
        raise ValueError(
            "model output reached the token limit before completing JSON; "
            "the response may contain a repetition loop"
        )

    result = parse_json_content(content)
    if isinstance(result, dict) and isinstance(result.get("clusters"), list):
        return result
    if isinstance(result, dict) and isinstance(result.get("assignments"), dict):
        labels = result.get("labels", {})
        grouped = {}
        for assignment_key, cluster_id in result["assignments"].items():
            paper_id = assignment_key
            if paper_ids is not None:
                try:
                    paper_id = paper_ids[int(assignment_key)]
                except (ValueError, IndexError):
                    raise ValueError(f"invalid record_index: {assignment_key}") from None
            cluster_id = str(cluster_id)
            grouped.setdefault(cluster_id, []).append(paper_id)
        return {
            "clusters": [
                {
                    "label": labels.get(cluster_id, f"cluster {cluster_id}"),
                    "paper_ids": group_ids,
                }
                for cluster_id, group_ids in grouped.items()
            ]
        }
    raise ValueError("output must contain clusters or assignments")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    response = json.loads(args.input.read_text())
    result = normalize_response(response)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
