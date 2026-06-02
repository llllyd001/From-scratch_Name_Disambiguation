import re

from .text_utils import clean_text


ORG_PATTERNS = [
    (r"\bvanderbilt\b", "Vanderbilt University"),
    (r"\bzhejiang university of technology\b|\bzhejiang univ(?:ersity)? technol\b", "Zhejiang University of Technology"),
    (r"\bjiangnan university\b|\bsouthern yangtze university\b", "Jiangnan University"),
    (r"\beast china normal university\b", "East China Normal University"),
    (r"\becnu\b", "East China Normal University"),
    (r"\bshanghai jiao ?tong university\b|\bshanghai jiaotong university\b", "Shanghai Jiao Tong University"),
    (r"\barizona state university\b", "Arizona State University"),
    (r"\bnanjing general hospital\b", "Nanjing General Hospital"),
    (r"\bnanjing red cross hospital\b", "Nanjing Red Cross Hospital"),
    (r"\bscripps research institute\b", "Scripps Research Institute"),
    (r"\bchinese academy of sciences\b", "Chinese Academy of Sciences"),
    (r"\buniversity of minnesota\b|\bminnesota university\b|\bminnesota univ\b", "University of Minnesota"),
    (r"\bcleveland state university\b", "Cleveland State University"),
    (r"\bibm\b", "IBM"),
]


def normalize_org(raw_org):
    text = clean_text(raw_org)
    if not text:
        return ""

    key = normalize_words(text)
    for pattern, institution in ORG_PATTERNS:
        if re.search(pattern, key):
            return institution

    match = re.search(
        r"([a-z][a-z ]{2,}?(?:university|hospital|institute|academy|company|corporation|laboratory|center))",
        key,
    )
    if match:
        return title_case_org(match.group(1))
    return text


def normalize_words(text):
    text = text.lower()
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"\belectronic address\b.*", " ", text)
    text = re.sub(r"\buniv\b|\buniv\.\b", "university", text)
    text = re.sub(r"\bdept\b|\bdept\.\b", "department", text)
    text = re.sub(r"\bmed ctr\b|\bmed(?:ical)? center\b", "medical center", text)
    text = re.sub(r"\bcanc ctr\b|\bcancer ctr\b", "cancer center", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return clean_text(text)


def title_case_org(text):
    small_words = {"of", "and", "for", "the"}
    words = [word if word in small_words else word.capitalize() for word in text.split()]
    return " ".join(words)
