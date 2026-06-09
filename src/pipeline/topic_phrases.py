import re
from collections import Counter

from .text_utils import clean_text


TOPIC_STOPWORDS = {
    "and", "or", "of", "in", "on", "for", "to", "with", "by", "from", "the",
    "a", "an", "based", "using", "study", "studies", "research", "analysis",
    "journal", "conference", "international", "proceedings", "letters",
}

GENERIC_TOPIC_WORDS = {
    "environmental", "food", "chemistry", "effects", "science", "model", "risk",
    "health", "medicine", "treatment", "expression", "association", "abstract",
    "women", "chinese", "rats", "rat", "survival", "cancer", "breast",
}


def build_specific_topic_candidates(papers, paper_ids, limit=220):
    phrase_counter = Counter()
    for paper_id in paper_ids:
        paper = papers[paper_id]
        for phrase in paper_topic_phrases(paper):
            phrase_counter[phrase] += 1

    phrases = [
        phrase
        for phrase, count in phrase_counter.most_common(limit * 2)
        if keep_phrase(phrase, count)
    ][:limit]
    return [
        {"id": f"specific_topic_{index:03d}", "canonical": phrase, "variants": [phrase]}
        for index, phrase in enumerate(phrases, start=1)
    ]


def paper_topic_phrases(paper):
    phrases = []
    for keyword in paper.get("keywords") or []:
        phrase = normalize_phrase(keyword)
        if phrase:
            phrases.append(phrase)

    title_tokens = content_tokens(paper.get("title", ""))
    venue_tokens = content_tokens(paper.get("venue", ""))
    phrases.extend(title_ngrams(title_tokens, min_n=2))
    phrases.extend(title_ngrams(venue_tokens, min_n=2, max_n=2))
    return phrases


def title_ngrams(tokens, min_n=1, max_n=3):
    phrases = []
    for n in range(min_n, max_n + 1):
        for index in range(len(tokens) - n + 1):
            phrase = " ".join(tokens[index:index + n])
            if phrase:
                phrases.append(phrase)
    return phrases


def content_tokens(text):
    return [
        token
        for token in re.findall(r"[a-z0-9]+", clean_text(text).lower())
        if len(token) > 2 and token not in TOPIC_STOPWORDS
    ]


def normalize_phrase(text):
    tokens = content_tokens(text)
    return " ".join(tokens)


def keep_phrase(phrase, count):
    tokens = phrase.split()
    if not phrase or len(tokens) == 0:
        return False
    if phrase in GENERIC_TOPIC_WORDS:
        return False
    if len(tokens) == 1 and count < 3:
        return False
    if len(tokens) >= 4 and count < 2:
        return False
    if all(token.isdigit() for token in tokens):
        return False
    return True
