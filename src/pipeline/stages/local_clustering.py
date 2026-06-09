from collections import defaultdict

from ..text_utils import norm


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


def build_local_clusters(profiles):
    identity_clusters = merge_by_identity(profiles)
    content_clusters = merge_by_content(identity_clusters, profiles)
    return identity_clusters, content_clusters


def merge_by_identity(profiles):
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

    for grouped_ids in org_index.values():
        union_all(union_find, grouped_ids)

    for grouped_ids in coauthor_index.values():
        for left_index, left_id in enumerate(grouped_ids):
            for right_id in grouped_ids[left_index + 1:]:
                if coauthor_compatible(profiles[left_id], profiles[right_id]):
                    union_find.union(left_id, right_id)

    return collect_clusters(union_find, paper_ids)


def merge_by_content(clusters, profiles):
    union_find = UnionFind(range(len(clusters)))
    summaries = [local_cluster_features(cluster, profiles) for cluster in clusters]

    for left_index, left in enumerate(summaries):
        for right_index, right in enumerate(summaries[left_index + 1:], start=left_index + 1):
            if content_compatible(left, right):
                union_find.union(left_index, right_index)

    merged = defaultdict(list)
    for index, cluster in enumerate(clusters):
        merged[union_find.find(index)].extend(cluster)
    return sorted(merged.values(), key=lambda ids: (-len(ids), ids[0]))


def content_compatible(left, right):
    if left["paper_count"] > 3 and right["paper_count"] > 3:
        return False

    shared_specific = overlap_count(left["specific_topics"], right["specific_topics"])
    shared_broad = overlap_count(left["broad_topics"], right["broad_topics"])
    shared_venues = overlap_count(left["venues"], right["venues"])
    shared_coauthors = overlap_count(left["coauthors"], right["coauthors"])

    if shared_specific >= 3 and shared_broad >= 1 and (shared_venues >= 1 or shared_coauthors >= 1):
        return True
    return False


def local_cluster_features(paper_ids, profiles):
    fields = {
        "coauthors": [],
        "venues": [],
        "broad_topics": [],
        "specific_topics": [],
    }
    for paper_id in paper_ids:
        profile = profiles[paper_id]
        identity = profile["identity_profile"]
        research = profile["research_profile"]
        fields["coauthors"].extend(identity["coauthors"])
        fields["venues"].extend([identity["venue"]] if identity["venue"] else [])
        fields["broad_topics"].extend(research["broad_topics"])
        fields["specific_topics"].extend(research["specific_topics"])
    return {
        "paper_count": len(paper_ids),
        **{field: set(values) for field, values in fields.items()},
    }


def summarize_cluster(cluster_id, paper_ids, profiles, candidates):
    orgs = []
    coauthors = []
    venues = []
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
        venues.extend([identity["venue"]] if identity["venue"] else [])
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
        "venues": names(venues, "venues", 5),
        "coauthors": names(coauthors, "coauthors", 8),
        "broad_topics": names(broad_topics, "broad_topics", 4),
        "specific_topics": names(specific_topics, "specific_topics", 5),
        "representative_papers": representative_papers(paper_ids, profiles),
    }


def representative_papers(paper_ids, profiles, limit=3):
    papers = []
    for paper_id in paper_ids[:limit]:
        profile = profiles[paper_id]
        raw = profile["raw_evidence"]
        papers.append({
            "paper_id": paper_id,
            "year": profile.get("year"),
            "author_position": profile.get("author_position"),
            "title": raw.get("title", "")[:180],
            "venue": raw.get("venue", ""),
            "keywords": raw.get("keywords", [])[:6],
            "organization": raw.get("organization", "")[:180],
        })
    return papers


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


def collect_clusters(union_find, paper_ids):
    clusters = defaultdict(list)
    for paper_id in paper_ids:
        clusters[union_find.find(paper_id)].append(paper_id)
    return sorted(clusters.values(), key=lambda ids: (-len(ids), ids[0]))


def union_all(union_find, grouped_ids):
    for paper_id in grouped_ids[1:]:
        union_find.union(grouped_ids[0], paper_id)


def is_valid_value(value):
    return norm(value) not in EMPTY_VALUES


def topic_compatible(left, right):
    left_topics = set(left["research_profile"]["broad_topics"])
    right_topics = set(right["research_profile"]["broad_topics"])
    return bool(left_topics & right_topics)


def coauthor_compatible(left, right):
    left_coauthors = set(left["identity_profile"]["coauthors"])
    right_coauthors = set(right["identity_profile"]["coauthors"])
    return len(left_coauthors & right_coauthors) >= 2 and topic_compatible(left, right)


def overlap_count(left, right):
    return len(set(left) & set(right))
