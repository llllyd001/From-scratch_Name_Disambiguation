def recover_merges_with_llm(llm, target_key, summaries, threshold):
    policy, policy_stats = choose_merge_policy(summaries)
    config = policy_config(policy)
    anchors = [cluster for cluster in summaries if cluster["paper_count"] >= config["min_anchor_size"]][:config["max_anchors"]]
    small_clusters = [cluster for cluster in summaries if cluster["paper_count"] <= 3]
    if not anchors or not small_clusters:
        return policy, policy_stats, []

    proposals = []
    for anchor in anchors:
        candidates = ranked_candidates(anchor, small_clusters, config)
        if candidates:
            proposals.extend(llm.suggest_anchor_merges(target_key, anchor, candidates, threshold))
    return policy, policy_stats, dedupe_merges(proposals)


def ranked_candidates(anchor, candidates, config):
    scored = []
    for candidate in candidates:
        if not eligible_candidate(anchor, candidate, config["policy"]):
            continue
        score = blocking_score(anchor, candidate)
        if score:
            scored.append((score, candidate["paper_count"], candidate))
    scored.sort(key=lambda item: (-item[0], item[1], item[2]["id"]))
    return [candidate for _, _, candidate in scored[:config["candidate_limit"]]]


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


def overlap_count(left, right):
    return len(set(left) & set(right))


def eligible_candidate(anchor, candidate, policy):
    shared_org = overlap_count(anchor["organizations"], candidate["organizations"])
    shared_coauthors = overlap_count(anchor["coauthors"], candidate["coauthors"])
    shared_venues = overlap_count(anchor.get("venues", []), candidate.get("venues", []))
    shared_specific = overlap_count(anchor["specific_topics"], candidate["specific_topics"])

    if shared_coauthors >= 1:
        return True
    if shared_org >= 1 and shared_specific >= 1:
        return True
    if policy == "permissive" and anchor["paper_count"] < 20 and shared_venues >= 1 and shared_specific >= 1:
        return True
    return False


def choose_merge_policy(summaries):
    paper_counts = sorted((cluster["paper_count"] for cluster in summaries), reverse=True)
    singleton_ratio = sum(count == 1 for count in paper_counts) / len(paper_counts) if paper_counts else 0
    top_anchors = summaries[: min(5, len(summaries))]
    evidence_coverage = average_anchor_evidence(top_anchors)
    pairwise_stats = anchor_pairwise_stats(top_anchors)

    stats = {
        "singleton_ratio": round(singleton_ratio, 3),
        "top_anchor_count": len(top_anchors),
        "top_anchor_evidence": round(evidence_coverage, 3),
        "pair_avg_shared_org": round(pairwise_stats["avg_shared_org"], 3),
        "pair_avg_shared_coauthors": round(pairwise_stats["avg_shared_coauthors"], 3),
        "pair_avg_shared_specific": round(pairwise_stats["avg_shared_specific"], 3),
        "pair_avg_shared_broad": round(pairwise_stats["avg_shared_broad"], 3),
        "pair_risk_score": round(pairwise_stats["risk_score"], 3),
    }

    if pairwise_stats["risk_score"] >= 1.55:
        return "conservative", stats
    if pairwise_stats["risk_score"] <= 0.85 and evidence_coverage >= 2.0:
        return "permissive", stats
    if singleton_ratio >= 0.75 and pairwise_stats["risk_score"] <= 1.2:
        return "permissive", stats
    return "conservative", stats


def policy_config(policy):
    if policy == "conservative":
        return {
            "policy": policy,
            "min_anchor_size": 3,
            "max_anchors": 12,
            "candidate_limit": 10,
        }
    return {
        "policy": policy,
        "min_anchor_size": 3,
        "max_anchors": 18,
        "candidate_limit": 14,
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


def dedupe_merges(merges):
    best = {}
    for merge in merges:
        key = tuple(sorted(merge["clusters"]))
        if key not in best or merge["confidence"] > best[key]["confidence"]:
            best[key] = merge
    return list(best.values())
