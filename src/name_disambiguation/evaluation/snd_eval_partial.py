import argparse
import json
from typing import Dict, Any

from name_disambiguation.data_io import load_json
from name_disambiguation.evaluation.snd_eval import pairwise_evaluate
from name_disambiguation.paths import dataset_dir


def load_partial_predict_result(predict_result: Any) -> Dict[str, Any]:
    if isinstance(predict_result, dict):
        return predict_result

    if not isinstance(predict_result, str):
        raise ValueError("predict_result must be a dict or file path")

    if predict_result.endswith(".jsonl"):
        merged_result = {}
        with open(predict_result, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at line {line_no}: {exc}") from exc

                name = obj.get("name")
                clusters = obj.get("clusters")
                if name is None or clusters is None:
                    raise ValueError(
                        f"Invalid record at line {line_no}: expected keys 'name' and 'clusters'"
                    )
                # If a name appears multiple times, keep the latest one.
                merged_result[name] = clusters
        return merged_result

    loaded = load_json(predict_result)
    if isinstance(loaded, dict):
        return loaded
    raise ValueError("Unsupported predict result format")


def evaluate_partial(predict_result, ground_truth):
    predict_result = load_partial_predict_result(predict_result)
    if isinstance(ground_truth, str):
        ground_truth = load_json(ground_truth)

    name_nums = 0
    result_list = []
    skipped_not_in_gt = 0
    skipped_incomplete = 0

    for name in predict_result:
        if name not in ground_truth:
            skipped_not_in_gt += 1
            continue

        predicted_pubs = {}
        for idx, pids in enumerate(predict_result[name]):
            for pid in pids:
                predicted_pubs[pid] = idx

        pubs = []
        ilabel = 0
        true_labels = []
        if isinstance(ground_truth[name], dict):
            for aid in ground_truth[name]:
                pubs.extend(ground_truth[name][aid])
                true_labels.extend([ilabel] * len(ground_truth[name][aid]))
                ilabel += 1
        else:
            for cluster in ground_truth[name]:
                pubs.extend(cluster)
                true_labels.extend([ilabel] * len(cluster))
                ilabel += 1

        if any(pid not in predicted_pubs for pid in pubs):
            skipped_incomplete += 1
            continue

        predict_labels = [predicted_pubs[pid] for pid in pubs]

        pairwise_precision, pairwise_recall, pairwise_f1 = pairwise_evaluate(true_labels, predict_labels)
        result_list.append((pairwise_precision, pairwise_recall, pairwise_f1))
        name_nums += 1

    if name_nums == 0:
        print("No valid names to evaluate in partial result.")
        print(f"Skipped names not in ground truth: {skipped_not_in_gt}")
        print(f"Skipped incomplete predicted names: {skipped_incomplete}")
        return 0.0

    avg_pairwise_f1 = sum([result[2] for result in result_list]) / name_nums
    print(f"Evaluated names: {name_nums}")
    print(f"Skipped names not in ground truth: {skipped_not_in_gt}")
    print(f"Skipped incomplete predicted names: {skipped_incomplete}")
    print(f"Current Partial Average Pairwise F1: {avg_pairwise_f1:.3f}")
    return avg_pairwise_f1


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate partial SND clustering result.")
    parser.add_argument(
        "--predict",
        type=str,
        required=True,
        help="Path to partial/full predict result file (.jsonl or .json).",
    )
    parser.add_argument(
        "--ground-truth",
        type=str,
        nargs="?",
        const=str(dataset_dir("v3") / "sna_valid_example.json"),
        default=str(dataset_dir("v3") / "sna_valid_example.json"),
        help="Path to ground truth json. If omitted after the flag, the default validation ground truth is used.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_partial(args.predict, args.ground_truth)
