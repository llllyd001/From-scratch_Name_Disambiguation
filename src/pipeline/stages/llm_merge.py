ANCHOR_MIN_PAPERS = 15
REPAIR_MAX_PAPERS = 14
CANDIDATE_BATCH_SIZE = 30
MAX_LARGE_ANCHORS = 12
MAX_REPAIR_ANCHORS = 24
REPAIR_CANDIDATE_LIMIT = 12


def recover_merges_with_llm(llm, target_key, summaries, threshold):
    policy = "strict_all_small"
    anchors = [cluster for cluster in summaries if cluster["paper_count"] >= ANCHOR_MIN_PAPERS][:MAX_LARGE_ANCHORS]
    repair_clusters = [cluster for cluster in summaries if cluster["paper_count"] <= REPAIR_MAX_PAPERS]
    policy_stats = merge_stats(summaries, anchors, repair_clusters)
    if not repair_clusters:
        return policy, policy_stats, []

    proposals = []
    for anchor in anchors:
        for batch in large_anchor_candidate_batches(anchor, repair_clusters):
            proposals.extend(llm.suggest_anchor_merges(target_key, anchor, batch, threshold))

    for anchor in repair_anchors(repair_clusters):
        candidates = ranked_repair_candidates(anchor, repair_clusters)
        if candidates:
            proposals.extend(llm.suggest_anchor_merges(target_key, anchor, candidates, threshold))

    return policy, policy_stats, dedupe_merges(proposals)


def large_anchor_candidate_batches(anchor, repair_clusters):
    candidates = [cluster for cluster in repair_clusters if cluster["id"] != anchor["id"]]
    if len(candidates) <= CANDIDATE_BATCH_SIZE:
        return [candidates] if candidates else []
    ranked = ranked_by_blocking_score(anchor, candidates, CANDIDATE_BATCH_SIZE)
    return [ranked] if ranked else []


def ranked_by_blocking_score(anchor, candidates, limit):
    scored = []
    for candidate in candidates:
        score = blocking_score(anchor, candidate)
        if score:
            scored.append((score, candidate["paper_count"], candidate))
    scored.sort(key=lambda item: (-item[0], item[1], item[2]["id"]))
    return [candidate for _, _, candidate in scored[:limit]]


def repair_anchors(repair_clusters):
    anchors = [cluster for cluster in repair_clusters if cluster["paper_count"] >= 2]
    anchors.sort(key=lambda cluster: (-cluster["paper_count"], cluster["id"]))
    return anchors[:MAX_REPAIR_ANCHORS]


def ranked_repair_candidates(anchor, repair_clusters):
    scored = []
    for candidate in repair_clusters:
        if cluster_number(candidate["id"]) <= cluster_number(anchor["id"]):
            continue
        if not repair_candidate(anchor, candidate):
            continue
        score = blocking_score(anchor, candidate)
        if score:
            scored.append((score, candidate["paper_count"], candidate))
    scored.sort(key=lambda item: (-item[0], item[1], item[2]["id"]))
    return [candidate for _, _, candidate in scored[:REPAIR_CANDIDATE_LIMIT]]


def blocking_score(left, right):
    shared_org = overlap_count(left["organizations"], right["organizations"])
    shared_coauthors = overlap_count(left["coauthors"], right["coauthors"])
    shared_venues = overlap_count(left.get("venues", []), right.get("venues", []))
    shared_broad = overlap_count(left["broad_topics"], right["broad_topics"])
    shared_specific = overlap_count(left["specific_topics"], right["specific_topics"])
    score = 0
    score += 5 * shared_coauthors
    score += 4 * shared_org
    score += 3 * shared_specific
    score += 2 * shared_venues
    score += shared_broad
    return score


def repair_candidate(anchor, candidate):
    shared_org = overlap_count(anchor["organizations"], candidate["organizations"])
    shared_coauthors = overlap_count(anchor["coauthors"], candidate["coauthors"])
    shared_venues = overlap_count(anchor.get("venues", []), candidate.get("venues", []))
    shared_broad = overlap_count(anchor["broad_topics"], candidate["broad_topics"])
    shared_specific = overlap_count(anchor["specific_topics"], candidate["specific_topics"])

    if shared_coauthors >= 1:
        return True
    if shared_org >= 1 and shared_specific >= 1:
        return True
    if shared_venues >= 1 and shared_specific >= 1:
        return True
    if shared_specific >= 2 and shared_broad >= 1:
        return True
    return False


def merge_stats(summaries, anchors, repair_clusters):
    paper_counts = sorted((cluster["paper_count"] for cluster in summaries), reverse=True)
    singleton_ratio = sum(count == 1 for count in paper_counts) / len(paper_counts) if paper_counts else 0
    top_anchors = anchors[: min(5, len(anchors))]
    pairwise_stats = anchor_pairwise_stats(top_anchors)
    return {
        "singleton_ratio": round(singleton_ratio, 3),
        "anchor_min_papers": ANCHOR_MIN_PAPERS,
        "repair_max_papers": REPAIR_MAX_PAPERS,
        "large_anchor_count": len(anchors),
        "repair_cluster_count": len(repair_clusters),
        "top_anchor_count": len(top_anchors),
        "top_anchor_evidence": round(average_anchor_evidence(top_anchors), 3),
        "pair_avg_shared_org": round(pairwise_stats["avg_shared_org"], 3),
        "pair_avg_shared_coauthors": round(pairwise_stats["avg_shared_coauthors"], 3),
        "pair_avg_shared_specific": round(pairwise_stats["avg_shared_specific"], 3),
        "pair_avg_shared_broad": round(pairwise_stats["avg_shared_broad"], 3),
        "pair_risk_score": round(pairwise_stats["risk_score"], 3),
    }


def average_anchor_evidence(anchors):
    if not anchors:
        return 0
    scores = []
    for anchor in anchors:
        score = 0
        score += bool(anchor["organizations"])
        score += bool(anchor["coauthors"])
        score += bool(anchor.get("venues"))
        score += bool(anchor["specific_topics"])
        scores.append(score)
    return sum(scores) / len(scores)


def anchor_pairwise_stats(anchors):
    if len(anchors) < 2:
        return {
            "avg_shared_org": 0,
            "avg_shared_coauthors": 0,
            "avg_shared_specific": 0,
            "avg_shared_broad": 0,
            "risk_score": 0,
        }

    pair_count = 0
    shared_org = 0
    shared_coauthors = 0
    shared_specific = 0
    shared_broad = 0
    for left_index, left in enumerate(anchors):
        for right in anchors[left_index + 1:]:
            pair_count += 1
            shared_org += overlap_count(left["organizations"], right["organizations"])
            shared_coauthors += overlap_count(left["coauthors"], right["coauthors"])
            shared_specific += overlap_count(left["specific_topics"], right["specific_topics"])
            shared_broad += overlap_count(left["broad_topics"], right["broad_topics"])

    avg_shared_org = shared_org / pair_count
    avg_shared_coauthors = shared_coauthors / pair_count
    avg_shared_specific = shared_specific / pair_count
    avg_shared_broad = shared_broad / pair_count
    risk_score = (
        3.5 * avg_shared_org
        + 5.0 * avg_shared_coauthors
        + 4.0 * avg_shared_specific
        + 0.25 * avg_shared_broad
    )
    return {
        "avg_shared_org": avg_shared_org,
        "avg_shared_coauthors": avg_shared_coauthors,
        "avg_shared_specific": avg_shared_specific,
        "avg_shared_broad": avg_shared_broad,
        "risk_score": risk_score,
    }


def overlap_count(left, right):
    return len(set(left) & set(right))


def cluster_number(cluster_id):
    digits = "".join(char for char in cluster_id if char.isdigit())
    return int(digits) if digits else 0


def dedupe_merges(merges):
    best = {}
    for merge in merges:
        key = tuple(sorted(merge["clusters"]))
        if key not in best or merge["confidence"] > best[key]["confidence"]:
            best[key] = merge
    return list(best.values())
