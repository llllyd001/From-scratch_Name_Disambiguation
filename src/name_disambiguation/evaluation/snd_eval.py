import argparse
import re
from name_disambiguation.data_io import load_json


def normalize_predict_result(predict_result):
    normalized = {}

    for name, clusters in predict_result.items():
        if clusters and isinstance(clusters[0], str):
            base_name = re.sub(r"_\d+$", "", name)
            normalized.setdefault(base_name, []).append(clusters)
        else:
            normalized[name] = clusters

    return normalized


def evaluate(predict_result,ground_truth, per_name=False):
    if isinstance(predict_result, str):
        predict_result = load_json(predict_result)
    if isinstance(ground_truth, str):
        ground_truth = load_json(ground_truth)
    predict_result = normalize_predict_result(predict_result)

    name_nums = 0
    result_list = []
    per_name_results = []
    for name in predict_result:
        if name not in ground_truth:
            print(f"Skip {name}: not found in ground truth")
            continue

        #Get clustering labels in predict_result
        predicted_pubs = dict()
        for idx,pids in enumerate(predict_result[name]):
            for pid in pids:
                predicted_pubs[pid] = idx
        # Get paper labels in ground_truth
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

        predict_labels = []
        missing_pids = []
        next_missing_label = len(set(predicted_pubs.values()))
        for pid in pubs:
            if pid in predicted_pubs:
                predict_labels.append(predicted_pubs[pid])
            else:
                missing_pids.append(pid)
                predict_labels.append(next_missing_label)
                next_missing_label += 1
        if missing_pids:
            print(f"Fill {name}: prediction misses {len(missing_pids)} papers as singleton clusters")

        pairwise_precision, pairwise_recall, pairwise_f1 = pairwise_evaluate(true_labels,predict_labels)
        result_list.append((pairwise_precision,pairwise_recall,pairwise_f1))
        per_name_results.append((name, pairwise_precision, pairwise_recall, pairwise_f1))
        name_nums += 1

    if name_nums == 0:
        print("No valid names to evaluate.")
        return 0.0

    if per_name:
        for name, pairwise_precision, pairwise_recall, pairwise_f1 in per_name_results:
            print(
                f"{name}: precision={pairwise_precision:.3f} "
                f"recall={pairwise_recall:.3f} f1={pairwise_f1:.3f}"
            )

    avg_pairwise_f1 = sum([result[2] for result in result_list])/name_nums
    print(f'Evaluated names: {name_nums}')
    print(f'Average Pairwise F1: {avg_pairwise_f1:.3f}')

    return avg_pairwise_f1



def pairwise_evaluate(correct_labels, pred_labels):
    TP = 0.0  # Pairs Correctly Predicted To SameAuthor
    TP_FP = 0.0  # Total Pairs Predicted To SameAuthor
    TP_FN = 0.0  # Total Pairs To SameAuthor

    for i in range(len(correct_labels)):
        for j in range(i + 1, len(correct_labels)):
            if correct_labels[i] == correct_labels[j]:
                TP_FN += 1
            if pred_labels[i] == pred_labels[j]:
                TP_FP += 1
            if (correct_labels[i] == correct_labels[j]) and (pred_labels[i] == pred_labels[j]):
                TP += 1

    if TP == 0:
        pairwise_precision = 0
        pairwise_recall = 0
        pairwise_f1 = 0
    else:
        pairwise_precision = TP / TP_FP
        pairwise_recall = TP / TP_FN
        pairwise_f1 = (2 * pairwise_precision * pairwise_recall) / (pairwise_precision + pairwise_recall)

    return pairwise_precision, pairwise_recall, pairwise_f1




def main():
    parser = argparse.ArgumentParser(description='Evaluate SND results using pairwise F1 metric')
    parser.add_argument('--predict', type=str, required=True, help='Path to prediction result JSON file')
    parser.add_argument('--ground-truth', type=str, required=True, help='Path to ground truth JSON file')
    parser.add_argument('--per-name', action='store_true', help='Print precision/recall/F1 for each evaluated name')
    args = parser.parse_args()

    evaluate(args.predict, args.ground_truth, per_name=args.per_name)


if __name__ == '__main__':
    main()
