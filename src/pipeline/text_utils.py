import re


def clean_text(text):
    return " ".join((text or "").split())


def norm(text):
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())


def compact_name(name):
    return re.sub(r"[^a-z]", "", (name or "").lower())


def is_target_author(author_name, target_key):
    parts = target_key.replace("_", " ").split()
    variants = {"".join(parts), "".join(reversed(parts))}
    return compact_name(author_name) in variants
