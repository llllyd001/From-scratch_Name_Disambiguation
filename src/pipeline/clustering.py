from collections import defaultdict

from .text_utils import norm


EMPTY_VALUES = {"", "none", "null", "unknown", "n/a", "na", "not available"}


class UnionFind:
    def __init__(self, items):
        self.parent = {item: item for item in items}

    def find(self, item):
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, a, b):
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a != root_b:
            self.parent[root_b] = root_a


def initial_profile_clusters(profiles):
    paper_ids = list(profiles)
    union_find = UnionFind(paper_ids)
    org_index = defaultdict(list)
    coauthor_index = defaultdict(list)

    for paper_id, profile in profiles.items():
        identity = profile["identity_profile"]
        if is_valid_value(identity["organization"]):
            org_index[identity["organization"]].append(paper_id)
        for coauthor in identity["coauthors"]:
            if is_valid_value(coauthor):
                coauthor_index[coauthor].append(paper_id)

    for grouped_ids in list(org_index.values()) + list(coauthor_index.values()):
        for paper_id in grouped_ids[1:]:
            union_find.union(grouped_ids[0], paper_id)

    clusters = defaultdict(list)
    for paper_id in paper_ids:
        clusters[union_find.find(paper_id)].append(paper_id)
    return sorted(clusters.values(), key=lambda ids: (-len(ids), ids[0]))


def is_valid_value(value):
    return norm(value) not in EMPTY_VALUES


def summarize_cluster(cluster_id, paper_ids, profiles, candidates):
    orgs = []
    coauthors = []
    broad_topics = []
    specific_topics = []
    years = []
    for paper_id in paper_ids:
        profile = profiles[paper_id]
        identity = profile["identity_profile"]
        research = profile["research_profile"]
        years.append(profile.get("year"))
        orgs.extend([identity["organization"]] if identity["organization"] else [])
        coauthors.extend(identity["coauthors"])
        broad_topics.extend(research["broad_topics"])
        specific_topics.extend(research["specific_topics"])

    def names(ids, field, limit):
        lookup = {item["id"]: item["canonical"] for item in candidates[field]}
        return [lookup[item_id] for item_id in dict.fromkeys(ids) if item_id in lookup][:limit]

    clean_years = [year for year in years if year]
    return {
        "id": cluster_id,
        "paper_ids": paper_ids,
        "paper_count": len(paper_ids),
        "years": [min(clean_years), max(clean_years)] if clean_years else [],
        "organizations": names(orgs, "organizations", 5),
        "coauthors": names(coauthors, "coauthors", 8),
        "broad_topics": names(broad_topics, "broad_topics", 4),
        "specific_topics": names(specific_topics, "specific_topics", 5),
    }


def apply_cluster_merges(clusters, merges):
    union_find = UnionFind([cluster["id"] for cluster in clusters])
    for merge in merges:
        ids = merge.get("clusters", [])
        if len(ids) < 2:
            continue
        for cluster_id in ids[1:]:
            union_find.union(ids[0], cluster_id)

    id_to_papers = {cluster["id"]: cluster["paper_ids"] for cluster in clusters}
    merged = defaultdict(list)
    for cluster in clusters:
        merged[union_find.find(cluster["id"])].extend(id_to_papers[cluster["id"]])
    return sorted(merged.values(), key=lambda ids: (-len(ids), ids[0]))
