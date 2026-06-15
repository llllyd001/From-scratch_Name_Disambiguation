from collections import Counter, defaultdict


def pairwise_metrics(truth_groups, predicted_groups, all_ids):
    truth_label = {pid: i for i, group in enumerate(truth_groups) for pid in group}
    predicted_label = {}
    for i, group in enumerate(predicted_groups):
        for pid in group:
            predicted_label.setdefault(pid, i)
    next_label = len(predicted_groups)
    for pid in all_ids:
        if pid not in predicted_label:
            predicted_label[pid] = next_label
            next_label += 1

    choose_two = lambda count: count * (count - 1) // 2
    truth_counts = Counter(truth_label[pid] for pid in all_ids)
    predicted_counts = Counter(predicted_label[pid] for pid in all_ids)
    intersections = Counter(
        (truth_label[pid], predicted_label[pid]) for pid in all_ids
    )
    true_pairs = sum(choose_two(count) for count in truth_counts.values())
    predicted_pairs = sum(choose_two(count) for count in predicted_counts.values())
    tp = sum(choose_two(count) for count in intersections.values())
    precision = tp / predicted_pairs if predicted_pairs else 0
    recall = tp / true_pairs if true_pairs else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    return {"precision": precision, "recall": recall, "f1": f1}


def analyze_result(clusters, paper_ids, truth):
    valid = set(paper_ids)
    truth_by_paper = {pid: aid for aid, ids in truth.items() for pid in ids}
    memberships = defaultdict(list)
    compositions = []
    within_duplicates = {}
    predicted_groups = []

    for index, cluster in enumerate(clusters):
        ids = cluster.get("paper_ids", [])
        counts = Counter(ids)
        unique_ids = set(ids)
        valid_ids = unique_ids & valid
        predicted_groups.append(valid_ids)
        composition = Counter(truth_by_paper[pid] for pid in valid_ids)
        for pid in unique_ids:
            memberships[pid].append(index)
        duplicates = sorted(pid for pid, count in counts.items() if count > 1)
        if duplicates:
            within_duplicates[str(index)] = duplicates
        compositions.append({
            "cluster_index": index,
            "llm_label": cluster.get("label", ""),
            "composition": dict(composition),
        })

    truth_groups = {author_id: set(ids) for author_id, ids in truth.items()}
    author_coverage = {}
    fully_covered = 0
    for author_id, real_papers in truth_groups.items():
        best_count = max((len(real_papers & group) for group in predicted_groups), default=0)
        is_complete = any(real_papers <= group for group in predicted_groups)
        fully_covered += is_complete
        author_coverage[author_id] = {
            "papers": len(real_papers),
            "best_cluster_papers": best_count,
            "best_cluster_ratio": best_count / len(real_papers),
            "fully_covered": is_complete,
        }

    returned = set(memberships)
    pairwise = pairwise_metrics(list(truth_groups.values()), predicted_groups, paper_ids)
    return {
        "llm_clusters": len(clusters),
        "returned_valid_unique_papers": len(returned & valid),
        "missing_paper_ids": sorted(valid - returned),
        "unknown_paper_ids": sorted(returned - valid),
        "cross_cluster_duplicates": {
            pid: indexes for pid, indexes in memberships.items() if len(indexes) > 1
        },
        "within_cluster_duplicates": within_duplicates,
        "pairwise_metrics_missing_as_singletons": pairwise,
        "complete_author_coverage": {
            "fully_covered_authors": fully_covered,
            "total_authors": len(truth_groups),
            "rate": fully_covered / len(truth_groups),
        },
        "author_coverage": author_coverage,
        "cluster_compositions": compositions,
    }
