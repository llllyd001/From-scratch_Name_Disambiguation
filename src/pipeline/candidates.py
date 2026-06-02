from collections import Counter

from .evidence import coauthor_names, specific_topic_text, target_org_key
from .text_utils import clean_text
from .topic_taxonomy import broad_topic_candidates


MAX_FIELD_VALUES = 220


def unique_by_frequency(values, limit=MAX_FIELD_VALUES):
    counter = Counter(clean_text(value) for value in values if clean_text(value))
    return [value for value, _ in counter.most_common(limit)]


def collect_candidates(llm, papers, paper_ids, target_key):
    org_values = unique_by_frequency(target_org_key(papers[pid], target_key) for pid in paper_ids)
    coauthor_values = unique_by_frequency(name for pid in paper_ids for name in coauthor_names(papers[pid], target_key))
    venue_values = unique_by_frequency(papers[pid].get("venue", "") for pid in paper_ids)
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
        "broad_topics": broad_topic_candidates(),
        "specific_topics": llm.extract_candidates(
            "specific research topic",
            specific_topic_values,
            "specific_topic",
            "Summarize specific research directions from titles, keywords, and abstracts. Keep labels informative but not too long.",
        ),
    }
