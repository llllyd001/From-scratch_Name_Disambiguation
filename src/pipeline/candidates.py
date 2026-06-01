from collections import Counter

from .evidence import broad_topic_text, coauthor_names, specific_topic_text, target_org
from .text_utils import clean_text


MAX_FIELD_VALUES = 220


def unique_by_frequency(values):
    counter = Counter(clean_text(value) for value in values if clean_text(value))
    return [value for value, _ in counter.most_common(MAX_FIELD_VALUES)]


def collect_candidates(llm, papers, paper_ids, target_key):
    org_values = unique_by_frequency(target_org(papers[pid], target_key) for pid in paper_ids)
    coauthor_values = unique_by_frequency(name for pid in paper_ids for name in coauthor_names(papers[pid], target_key))
    venue_values = unique_by_frequency(papers[pid].get("venue", "") for pid in paper_ids)
    broad_topic_values = unique_by_frequency(broad_topic_text(papers[pid]) for pid in paper_ids)
    specific_topic_values = unique_by_frequency(specific_topic_text(papers[pid]) for pid in paper_ids)

    return {
        "organizations": llm.extract_candidates(
            "organization",
            org_values,
            "org",
            "Cluster equivalent institution, department, lab, or address strings. Be broad for spelling variants, but do not merge unrelated institutions.",
        ),
        "coauthors": llm.extract_candidates(
            "coauthor name",
            coauthor_values,
            "coauthor",
            "Cluster name variants for the same person, including name order changes, initials, hyphens, and capitalization variants.",
        ),
        "venues": llm.extract_candidates(
            "venue",
            venue_values,
            "venue",
            "Cluster equivalent journal or conference venue names. Keep different venues separate.",
        ),
        "broad_topics": llm.extract_candidates(
            "broad research area",
            broad_topic_values,
            "broad_topic",
            "Summarize broad disciplinary research areas from titles, venues, and abstracts.",
        ),
        "specific_topics": llm.extract_candidates(
            "specific research topic",
            specific_topic_values,
            "specific_topic",
            "Summarize specific research directions from titles, keywords, and abstracts. Keep labels informative but not too long.",
        ),
    }
