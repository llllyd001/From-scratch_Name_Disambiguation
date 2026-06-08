import math
import re
from collections import Counter, defaultdict

from .text_utils import clean_text


WEAK_WORDS = {
    "university", "department", "school", "college", "institute", "laboratory",
    "lab", "center", "centre", "faculty", "division", "key", "state", "national",
    "research", "medical", "hospital", "academy", "science", "sciences",
}


def normalize_org(raw_org):
    text = normalize_words(raw_org)
    return title_case_org(text) if text else ""


def normalize_words(text):
    text = (text or "").lower()
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"\belectronic address\b.*", " ", text)
    text = re.sub(r"\buniv\b|\buniv\.\b", "university", text)
    text = re.sub(r"\bdept\b|\bdept\.\b", "department", text)
    text = re.sub(r"\bsch\b|\bsch\.\b", "school", text)
    text = re.sub(r"\binst\b|\binst\.\b", "institute", text)
    text = re.sub(r"\blab\b|\blab\.\b", "laboratory", text)
    text = re.sub(r"\bmed ctr\b|\bmed(?:ical)? center\b", "medical center", text)
    text = re.sub(r"\bcanc ctr\b|\bcancer ctr\b", "cancer center", text)
    text = re.sub(r"\bpr china\b|\bpeople s republic of china\b", "china", text)
    text = re.sub(r"\b\d{5,6}\b", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return clean_text(text)


def org_similarity_key(org_text):
    tokens = [token for token in normalize_words(org_text).split() if token not in WEAK_WORDS]
    return " ".join(tokens) or normalize_words(org_text)


def build_org_candidates(raw_values, threshold=0.68):
    values = [value for value in dict.fromkeys(normalize_org(value) for value in raw_values) if value]
    if not values:
        return []

    keys = [org_similarity_key(value) for value in values]
    vectors = tfidf_vectors(keys)
    parent = {index: index for index in range(len(values))}

    for left in range(len(values)):
        for right in range(left + 1, len(values)):
            if cosine(vectors[left], vectors[right]) >= threshold:
                union(parent, left, right)

    groups = defaultdict(list)
    for index, value in enumerate(values):
        groups[find(parent, index)].append(value)

    candidates = []
    for group in sorted(groups.values(), key=lambda items: values.index(items[0])):
        canonical = min(group, key=len)
        candidates.append({
            "id": f"org_{len(candidates) + 1:03d}",
            "canonical": canonical,
            "variants": group,
        })
    return candidates


def char_ngrams(text):
    compact = re.sub(r"\s+", " ", f" {text} ")
    grams = []
    for size in (3, 4, 5):
        grams.extend(compact[index:index + size] for index in range(len(compact) - size + 1))
    return grams


def tfidf_vectors(texts):
    term_counts = [Counter(char_ngrams(text)) for text in texts]
    document_frequency = Counter(term for counts in term_counts for term in counts)
    vectors = []
    for counts in term_counts:
        vector = {}
        for term, count in counts.items():
            vector[term] = count * (math.log((len(texts) + 1) / (document_frequency[term] + 1)) + 1)
        vectors.append(vector)
    return vectors


def cosine(left, right):
    if not left or not right:
        return 0
    shared = set(left) & set(right)
    numerator = sum(left[term] * right[term] for term in shared)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    return numerator / (left_norm * right_norm) if left_norm and right_norm else 0


def find(parent, item):
    while parent[item] != item:
        parent[item] = parent[parent[item]]
        item = parent[item]
    return item


def union(parent, left, right):
    root_left = find(parent, left)
    root_right = find(parent, right)
    if root_left != root_right:
        parent[root_right] = root_left


def title_case_org(text):
    small_words = {"of", "and", "for", "the"}
    words = [word if word in small_words else word.capitalize() for word in text.split()]
    return " ".join(words)
