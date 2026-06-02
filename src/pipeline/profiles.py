import re

from .evidence import author_position, broad_topic_text, coauthor_names, missing_fields, specific_topic_text, target_org, target_org_key
from .text_utils import clean_text, norm


TOPIC_STOPWORDS = {
    "and", "or", "of", "in", "on", "for", "to", "with", "by", "from", "the",
    "a", "an", "based", "using", "study", "studies", "research", "analysis",
}


def build_variant_map(candidates):
    mapping = {}
    for candidate in candidates:
        for variant in candidate.get("variants", []):
            key = norm(variant)
            if key:
                mapping[key] = candidate["id"]
    return mapping


def match_one(value, candidates):
    key = norm(value)
    if not key:
        return None
    mapping = build_variant_map(candidates)
    if key in mapping:
        return mapping[key]
    for variant_key, candidate_id in mapping.items():
        if len(variant_key) > 10 and (variant_key in key or key in variant_key):
            return candidate_id
    return None


def match_many(values, candidates, limit=3):
    matched = []
    for value in values:
        candidate_id = match_one(value, candidates)
        if candidate_id and candidate_id not in matched:
            matched.append(candidate_id)
        if len(matched) >= limit:
            break
    return matched


def match_topics(text, candidates, limit=3):
    text_key = norm(text)
    text_tokens = topic_tokens(text)
    matched = []
    for candidate in candidates:
        values = [candidate.get("canonical", "")] + candidate.get("variants", [])
        if any(norm(value) and norm(value) in text_key for value in values):
            matched.append(candidate["id"])
        if len(matched) >= limit:
            break
    if matched:
        return matched

    scored = []
    for candidate in candidates:
        candidate_tokens = set()
        for value in [candidate.get("canonical", "")] + candidate.get("variants", []):
            candidate_tokens.update(topic_tokens(value))
        score = len(text_tokens & candidate_tokens)
        if score:
            scored.append((score, candidate["id"]))
    return [candidate_id for _, candidate_id in sorted(scored, reverse=True)[:limit]]


def topic_tokens(text):
    return {
        token
        for token in re.findall(r"[a-z0-9]+", (text or "").lower())
        if len(token) > 2 and token not in TOPIC_STOPWORDS
    }


def build_paper_profile(paper_id, paper, target_key, candidates):
    coauthors = coauthor_names(paper, target_key)
    return {
        "paper_id": paper_id,
        "target_name": target_key,
        "year": paper.get("year"),
        "author_position": author_position(paper, target_key),
        "identity_profile": {
            "organization": match_one(target_org_key(paper, target_key), candidates["organizations"]),
            "coauthors": match_many(coauthors, candidates["coauthors"], limit=3),
            "venue": match_one(paper.get("venue", ""), candidates["venues"]),
        },
        "research_profile": {
            "broad_topics": match_topics(broad_topic_text(paper), candidates["broad_topics"], limit=1),
            "specific_topics": match_topics(specific_topic_text(paper), candidates["specific_topics"], limit=3),
        },
        "raw_evidence": {
            "organization": target_org(paper, target_key),
            "organization_key": target_org_key(paper, target_key),
            "coauthors": coauthors[:8],
            "venue": clean_text(paper.get("venue", "")),
            "title": clean_text(paper.get("title", "")),
            "keywords": paper.get("keywords") or [],
        },
        "missing_fields": missing_fields(paper, target_key),
    }


def fill_missing_broad_topics(llm, profiles, candidates):
    missing = [
        {
            "paper_id": paper_id,
            "text": " | ".join([
                profile["raw_evidence"]["title"],
                profile["raw_evidence"]["venue"],
                ", ".join(profile["raw_evidence"]["keywords"]),
            ]),
        }
        for paper_id, profile in profiles.items()
        if not profile["research_profile"]["broad_topics"]
    ]
    if not missing:
        return

    assignments = llm.assign_broad_topics(missing, candidates["broad_topics"])
    fallback = candidates["broad_topics"][0]["id"] if candidates["broad_topics"] else None
    for item in missing:
        topic_id = assignments.get(item["paper_id"]) or fallback
        if topic_id:
            profiles[item["paper_id"]]["research_profile"]["broad_topics"] = [topic_id]
