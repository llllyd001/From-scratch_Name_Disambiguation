from collections import defaultdict

from features import cosine, metadata_score, vectorize


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


def cluster_one_target(papers, paper_ids, target_key, threshold, llm_judge):
    vectors, norms = vectorize(papers, paper_ids)
    union_find = UnionFind(paper_ids)

    for i, paper_id_a in enumerate(paper_ids):
        for j in range(i + 1, len(paper_ids)):
            paper_id_b = paper_ids[j]

            llm_answer = llm_judge.same_author(
                papers[paper_id_a],
                papers[paper_id_b],
                target_key,
            )
            if llm_answer is True:
                union_find.union(paper_id_a, paper_id_b)
                continue
            if llm_answer is False:
                continue

            text_score = cosine(vectors[i], norms[i], vectors[j], norms[j])
            score = 0.75 * text_score
            score += metadata_score(papers[paper_id_a], papers[paper_id_b], target_key)

            if score >= threshold:
                union_find.union(paper_id_a, paper_id_b)

    groups = defaultdict(list)
    for paper_id in paper_ids:
        groups[union_find.find(paper_id)].append(paper_id)

    return sorted(groups.values(), key=lambda ids: (-len(ids), ids[0]))
