from .text_utils import clean_text, is_target_author


def target_author_index(paper, target_key):
    for index, author in enumerate(paper.get("authors", [])):
        if is_target_author(author.get("name", ""), target_key):
            return index
    return None


def target_org(paper, target_key):
    index = target_author_index(paper, target_key)
    if index is None:
        return ""
    return clean_text(paper.get("authors", [])[index].get("org", ""))


def coauthor_names(paper, target_key):
    names = []
    for author in paper.get("authors", []):
        name = clean_text(author.get("name", ""))
        if name and not is_target_author(name, target_key):
            names.append(name)
    return names


def author_position(paper, target_key):
    index = target_author_index(paper, target_key)
    total = len(paper.get("authors", []))
    if index is None or total == 0:
        return {"index": None, "total": total, "role": "unknown"}
    if index == 0:
        role = "first"
    elif index == total - 1:
        role = "last"
    else:
        role = "middle"
    return {"index": index + 1, "total": total, "role": role}


def missing_fields(paper, target_key):
    missing = []
    if not target_org(paper, target_key):
        missing.append("target_org")
    for field in ["title", "abstract", "keywords", "venue", "year"]:
        if not paper.get(field):
            missing.append(field)
    return missing


def broad_topic_text(paper):
    parts = [
        clean_text(paper.get("title", "")),
        clean_text(paper.get("venue", "")),
        clean_text(paper.get("abstract", ""))[:250],
    ]
    return " | ".join(part for part in parts if part)


def specific_topic_text(paper):
    keywords = ", ".join(paper.get("keywords") or [])
    parts = [
        clean_text(paper.get("title", "")),
        keywords,
        clean_text(paper.get("abstract", ""))[:350],
    ]
    return " | ".join(part for part in parts if part)
