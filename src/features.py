import math
import re
from collections import Counter


STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "are", "was", "were",
    "into", "using", "used", "study", "analysis", "based", "between", "among",
    "through", "their", "have", "has", "had", "not", "can", "may", "our",
    "its", "in", "on", "of", "to", "a", "an", "by", "as", "at", "is", "be",
}


def compact_name(name):
    return re.sub(r"[^a-z]", "", name.lower())


def is_target_author(author_name, target_key):
    parts = target_key.replace("_", " ").split()
    variants = {"".join(parts), "".join(reversed(parts))}
    return compact_name(author_name) in variants


def words(text):
    return [
        word
        for word in re.findall(r"[a-zA-Z][a-zA-Z0-9]+", (text or "").lower())
        if len(word) > 1 and word not in STOPWORDS
    ]


def org_words(text):
    common_org_words = {"china", "chinese", "university", "college", "school"}
    return set(words(text)) - common_org_words


def paper_text(paper):
    fields = [
        paper.get("title", ""),
        paper.get("abstract", ""),
        " ".join(paper.get("keywords") or []),
        paper.get("venue", ""),
        str(paper.get("year", "")),
    ]

    for author in paper.get("authors", []):
        fields.append(author.get("name", ""))
        fields.append(author.get("org", ""))

    return " ".join(fields)


def target_org(paper, target_key):
    for author in paper.get("authors", []):
        if is_target_author(author.get("name", ""), target_key):
            return author.get("org", "")
    return ""


def coauthors(paper, target_key):
    names = set()
    for author in paper.get("authors", []):
        name = author.get("name", "")
        if name and not is_target_author(name, target_key):
            names.add(compact_name(name))
    return names


def jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def vectorize(papers, paper_ids):
    docs = [Counter(words(paper_text(papers[paper_id]))) for paper_id in paper_ids]
    doc_freq = Counter()

    for doc in docs:
        doc_freq.update(doc.keys())

    vectors = []
    norms = []
    total_docs = len(docs)

    for doc in docs:
        vector = {}
        for term, term_count in doc.items():
            if doc_freq[term] < 2:
                continue
            tf = 1.0 + math.log(term_count)
            idf = math.log((total_docs + 1) / (doc_freq[term] + 1)) + 1.0
            vector[term] = tf * idf

        norm = math.sqrt(sum(value * value for value in vector.values())) or 1.0
        vectors.append(vector)
        norms.append(norm)

    return vectors, norms


def cosine(vector_a, norm_a, vector_b, norm_b):
    if len(vector_a) > len(vector_b):
        vector_a, vector_b = vector_b, vector_a
        norm_a, norm_b = norm_b, norm_a

    dot = sum(value * vector_b.get(term, 0.0) for term, value in vector_a.items())
    return dot / (norm_a * norm_b)


def metadata_score(paper_a, paper_b, target_key):
    org_sim = jaccard(
        org_words(target_org(paper_a, target_key)),
        org_words(target_org(paper_b, target_key)),
    )
    coauthor_sim = jaccard(
        coauthors(paper_a, target_key),
        coauthors(paper_b, target_key),
    )

    year_a = paper_a.get("year")
    year_b = paper_b.get("year")
    year_bonus = 0.0
    if isinstance(year_a, int) and isinstance(year_b, int) and abs(year_a - year_b) <= 8:
        year_bonus = 0.05

    return 0.35 * org_sim + 0.45 * coauthor_sim + year_bonus
