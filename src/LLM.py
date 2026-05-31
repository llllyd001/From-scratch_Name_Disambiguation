import importlib
import re


DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


class LocalLLMJudge:
    """Local Hugging Face pairwise judge.

    Small local models are unstable when asked to output a whole JSON
    clustering result.  This class only asks a simpler YES/NO question for two
    papers at a time, then cluster.py handles the clustering logic.
    """

    def __init__(self, model_name=DEFAULT_MODEL_NAME, max_new_tokens=8, enabled=False):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.enabled = enabled
        self._pipeline = None

    def _load_pipeline(self):
        if self._pipeline is not None:
            return

        transformers = importlib.import_module("transformers")
        self._pipeline = transformers.pipeline(
            "text-generation",
            model=self.model_name,
            tokenizer=self.model_name,
        )

    def _paper_text(self, paper):
        authors = paper.get("authors") or []
        author_names = ", ".join(
            author["name"].strip()
            for author in authors
            if author.get("name")
        )
        return (
            f"Title: {paper.get('title', '').strip()}\n"
            f"Authors: {author_names}\n"
            f"Venue: {paper.get('venue', '').strip()}\n"
            f"Year: {paper.get('year', '')}\n"
            f"Keywords: {', '.join(paper.get('keywords') or [])}\n"
            f"Abstract: {paper.get('abstract', '').strip()[:1000]}"
        )

    def _build_prompt(self, paper_a, paper_b, target_name):
        return (
            f"Target author name: {target_name}\n"
            "Decide whether the target author in Paper A and Paper B is the same real person.\n"
            "Answer with exactly one word: YES or NO.\n\n"
            f"Paper A:\n{self._paper_text(paper_a)}\n\n"
            f"Paper B:\n{self._paper_text(paper_b)}\n\n"
            "Answer:"
        )

    def _parse_yes_no(self, text):
        match = re.search(r"\b(YES|NO)\b", text.upper())
        if match is None:
            return None
        if match.group(1) == "YES":
            return True
        return False

    def same_author(self, paper_a, paper_b, target_name):
        if not self.enabled:
            return None

        self._load_pipeline()
        prompt = self._build_prompt(paper_a, paper_b, target_name)
        result = self._pipeline(
            prompt,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            return_full_text=False,
        )
        return self._parse_yes_no(result[0]["generated_text"])
