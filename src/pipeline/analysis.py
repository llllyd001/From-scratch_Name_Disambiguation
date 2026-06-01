from collections import Counter


def build_truth_lookup(ground_truth, target_key):
    truth = ground_truth.get(target_key) if ground_truth else None
    lookup = {}
    if isinstance(truth, dict):
        for true_index, (true_key, paper_ids) in enumerate(truth.items()):
            for paper_id in paper_ids:
                lookup[paper_id] = {
                    "true_author_index": true_index,
                    "true_author_key": true_key,
                }
    elif isinstance(truth, list):
        for true_index, paper_ids in enumerate(truth):
            true_key = f"{target_key}_true_{true_index:04d}"
            for paper_id in paper_ids:
                lookup[paper_id] = {
                    "true_author_index": true_index,
                    "true_author_key": true_key,
                }
    return lookup


def true_author_summary(group, truth_lookup):
    counts = Counter(
        truth_lookup.get(paper_id, {}).get("true_author_key", "unknown")
        for paper_id in group
    )
    return [
        {"true_author_key": key, "paper_count": count}
        for key, count in counts.most_common()
    ]


def build_cluster_analysis(target_key, groups, papers, profiles, ground_truth=None):
    truth_lookup = build_truth_lookup(ground_truth, target_key)
    analysis = {}
    for group_index, group in enumerate(groups):
        cluster_key = f"{target_key}_{group_index:04d}"
        analysis[cluster_key] = {
            "paper_count": len(group),
            "true_author_summary": true_author_summary(group, truth_lookup),
            "papers": [
                profiles[paper_id]
                | truth_lookup.get(paper_id, {})
                | {"title": papers[paper_id].get("title", "")}
                for paper_id in group
            ],
        }
    return analysis
