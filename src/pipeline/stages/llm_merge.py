ANCHOR_MIN_PAPERS = 15
REPAIR_MAX_PAPERS = 14
MAX_ANCHORS_PER_REPAIR = 5
MAX_ANCHOR_PAPERS = 5
MAX_REPAIR_PAPERS = 3
PAIR_SCORE_THRESHOLD = 0.11

FEATURE_SETS = [
    ("coauthors_title_organization", ["coauthors", "title", "organization"]),
    ("title_organization", ["title", "organization"]),
    ("title_venue_organization", ["title", "venue", "organization"]),
    ("organization_venue", ["organization", "venue"]),
    ("coauthors_venue", ["coauthors", "venue"]),
    ("organization_keywords", ["organization", "keywords"]),
    ("coauthors_keywords", ["coauthors", "keywords"]),
]


def recover_merges_with_llm(llm, target_key, summaries, threshold, profiles=None):
    policy = "pairwise_vote"
    anchors = [cluster for cluster in summaries if cluster["paper_count"] >= ANCHOR_MIN_PAPERS]
    repair_clusters = [cluster for cluster in summaries if cluster["paper_count"] <= REPAIR_MAX_PAPERS]
    stats = merge_stats(summaries, anchors, repair_clusters, threshold)
    if not profiles or not anchors or not repair_clusters:
        return policy, stats, []

    proposals = []
    for repair in repair_clusters:
        for anchor in ranked_anchors(repair, anchors):
            proposal = pairwise_vote_merge(llm, target_key, anchor, repair, profiles, threshold)
            if proposal:
                proposals.append(proposal)
                break
    return policy, stats, proposals


def ranked_anchors(repair, anchors):
    scored = [(blocking_score(anchor, repair), anchor["paper_count"], anchor) for anchor in anchors]
    scored.sort(key=lambda item: (-item[0], -item[1], item[2]["id"]))
    return [anchor for _, _, anchor in scored[:MAX_ANCHORS_PER_REPAIR]]


def pairwise_vote_merge(llm, target_key, anchor, repair, profiles, vote_threshold):
    votes = []
    repair_papers = representative_papers(repair, profiles, MAX_REPAIR_PAPERS)
    anchor_papers = relevant_anchor_papers(anchor, repair, profiles, MAX_ANCHOR_PAPERS)
    for repair_paper in repair_papers:
        for anchor_paper in anchor_papers:
            feature_name, fields = best_feature_set(repair_paper, anchor_paper)
            if not fields:
                continue
            result = llm.compare_papers(
                target_key,
                repair_paper,
                anchor_paper,
                fields,
                PAIR_SCORE_THRESHOLD,
            )
            votes.append({
                "repair_paper": repair_paper["paper_id"],
                "anchor_paper": anchor_paper["paper_id"],
                "feature_set": feature_name,
                "same_author": result["same_author"],
                "score": round(result["same_author_score"], 3),
                "reason": result["reason"],
            })

    if not votes:
        return None
    positive = [vote for vote in votes if vote["same_author"]]
    vote_ratio = len(positive) / len(votes)
    if vote_ratio < vote_threshold:
        return None
    return {
        "clusters": [anchor["id"], repair["id"]],
        "confidence": round(vote_ratio, 3),
        "reason": f"{len(positive)}/{len(votes)} pairwise votes passed",
        "pair_score_avg": round(sum(vote["score"] for vote in votes) / len(votes), 3),
        "positive_votes": len(positive),
        "total_votes": len(votes),
        "votes": votes,
    }


def representative_papers(cluster, profiles, limit):
    paper_ids = cluster["paper_ids"][:limit]
    return [raw_paper_view(paper_id, profiles[paper_id]) for paper_id in paper_ids]


def relevant_anchor_papers(anchor, repair, profiles, limit):
    query_tokens = cluster_tokens(repair, profiles)
    scored = []
    for paper_id in anchor["paper_ids"]:
        score = len(query_tokens & paper_tokens(raw_paper_view(paper_id, profiles[paper_id])))
        scored.append((score, paper_id))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [raw_paper_view(paper_id, profiles[paper_id]) for _, paper_id in scored[:limit]]


def raw_paper_view(paper_id, profile):
    raw = profile["raw_evidence"]
    return {
        "paper_id": paper_id,
        "title": raw.get("title", ""),
        "keywords": raw.get("keywords", []),
        "venue": raw.get("venue", ""),
        "organization": raw.get("organization", ""),
        "coauthors": raw.get("coauthors", []),
    }


def best_feature_set(left, right):
    for name, fields in FEATURE_SETS:
        if all(has_field(left, field) and has_field(right, field) for field in fields):
            return name, fields
    return None, []


def has_field(paper, field):
    value = paper.get(field)
    if isinstance(value, list):
        return bool([item for item in value if item])
    return bool(value)


def blocking_score(anchor, repair):
    score = 0
    score += 5 * overlap_count(anchor["coauthors"], repair["coauthors"])
    score += 4 * overlap_count(anchor["organizations"], repair["organizations"])
    score += 3 * overlap_count(anchor["specific_topics"], repair["specific_topics"])
    score += 2 * overlap_count(anchor.get("venues", []), repair.get("venues", []))
    score += overlap_count(anchor["broad_topics"], repair["broad_topics"])
    return score


def cluster_tokens(cluster, profiles):
    tokens = set()
    for paper_id in cluster["paper_ids"]:
        tokens.update(paper_tokens(raw_paper_view(paper_id, profiles[paper_id])))
    return tokens


def paper_tokens(paper):
    tokens = set()
    for field in ["title", "venue", "organization"]:
        tokens.update(text_tokens(paper.get(field, "")))
    for keyword in paper.get("keywords", []):
        tokens.update(text_tokens(keyword))
    for coauthor in paper.get("coauthors", []):
        tokens.update(text_tokens(coauthor))
    return tokens


def text_tokens(text):
    return {
        token
        for token in "".join(char.lower() if char.isalnum() else " " for char in str(text)).split()
        if len(token) > 2
    }


def merge_stats(summaries, anchors, repair_clusters, threshold):
    counts = [cluster["paper_count"] for cluster in summaries]
    return {
        "vote_threshold": threshold,
        "pair_score_threshold": PAIR_SCORE_THRESHOLD,
        "anchor_min_papers": ANCHOR_MIN_PAPERS,
        "repair_max_papers": REPAIR_MAX_PAPERS,
        "large_anchor_count": len(anchors),
        "repair_cluster_count": len(repair_clusters),
        "singleton_ratio": round(sum(count == 1 for count in counts) / len(counts), 3) if counts else 0,
        "feature_priority": [name for name, _ in FEATURE_SETS],
    }


def overlap_count(left, right):
    return len(set(left) & set(right))
