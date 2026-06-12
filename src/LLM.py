import json
import os
import time
import urllib.error
import urllib.request


DEFAULT_MODEL_NAME = "deepseek-chat"
DEFAULT_BASE_URL = "https://models.sjtu.edu.cn/api/v1"


class CandidateLLM:
    def __init__(self, model_name=DEFAULT_MODEL_NAME, max_tokens=2048, max_retries=5, retry_delay=30):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.base_url = os.environ.get("ZHIYUAN_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
        self.api_key = os.environ.get("ZHIYUAN_API_KEY", "")

    def _chat(self, prompt):
        if not self.api_key:
            raise RuntimeError("ZHIYUAN_API_KEY is not set")

        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": self.max_tokens,
        }
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    data = json.loads(response.read().decode("utf-8"))
                return data["choices"][0]["message"]["content"]
            except urllib.error.HTTPError as error:
                if error.code != 429 or attempt >= self.max_retries:
                    raise
                wait_seconds = self.retry_delay * (attempt + 1)
                print(f"Rate limited by remote LLM, retrying in {wait_seconds}s", flush=True)
                time.sleep(wait_seconds)

    def _extract_json(self, text):
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return None

    def _fallback_candidates(self, prefix, values):
        return [
            {"id": f"{prefix}_{index:03d}", "canonical": value, "variants": [value]}
            for index, value in enumerate(values, start=1)
        ]

    def extract_candidates(self, field_name, values, prefix, instruction):
        values = [value for value in values if value]
        if not values:
            return []

        lines = [
            f"Build candidate set for field: {field_name}.",
            instruction,
            'Return JSON only: {"candidates":[{"canonical":"...","variants":["..."]}]}',
            "Rules:",
            "- group equivalent or near-equivalent values together",
            "- keep distinct people/institutions/topics separate",
            "- do not invent values not supported by the input",
            "- no explanation, no markdown",
            "",
            "Input values:",
        ]
        for index, value in enumerate(values, start=1):
            lines.append(f"v{index}: {value}")

        response = self._chat("\n".join(lines))
        parsed = self._extract_json(response)
        if not parsed or not isinstance(parsed.get("candidates"), list):
            return self._fallback_candidates(prefix, values)

        candidates = []
        for item in parsed["candidates"]:
            canonical = str(item.get("canonical", "")).strip()
            variants = [str(value).strip() for value in item.get("variants", []) if str(value).strip()]
            if not canonical:
                continue
            if canonical not in variants:
                variants.insert(0, canonical)
            candidates.append({
                "id": f"{prefix}_{len(candidates) + 1:03d}",
                "canonical": canonical,
                "variants": variants,
            })
        return candidates or self._fallback_candidates(prefix, values)

    def compare_papers(self, target_name, paper_a, paper_b, fields, score_threshold):
        prompt = "\n".join([
            f"Ambiguous author name: {target_name}",
            "Decide whether Paper A and Paper B were written by the same real person with this ambiguous name.",
            "Use only the fields shown below.",
            "Missing fields are unknown, not negative evidence.",
            "Give same_author_score from 0 to 1. Positive pairs should score higher than negative pairs.",
            f"Use same_author=true when same_author_score >= {score_threshold}.",
            'Return JSON only: {"same_author":true,"same_author_score":0.82,"confidence":0.82,"reason":"short reason"}',
            "",
            "Paper A:",
            self._format_paper_fields(paper_a, fields),
            "",
            "Paper B:",
            self._format_paper_fields(paper_b, fields),
        ])
        parsed = self._extract_json(self._chat(prompt)) or {}
        score = self._safe_score(parsed.get("same_author_score"), parsed.get("same_author"), parsed.get("confidence"))
        return {
            "same_author": score >= score_threshold,
            "same_author_score": score,
            "confidence": self._safe_float(parsed.get("confidence")),
            "reason": str(parsed.get("reason", "")),
        }

    def _format_paper_fields(self, paper, fields):
        lines = []
        for field in fields:
            value = paper.get(field, "")
            if isinstance(value, list):
                value = "; ".join(str(item) for item in value if item)
            lines.append(f"{field}: {value if value else '[missing]'}")
        return "\n".join(lines)

    def _safe_float(self, value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def _safe_score(self, value, same_author, confidence):
        if value is not None:
            score = self._safe_float(value)
        else:
            confidence = self._safe_float(confidence)
            score = confidence if same_author else 1 - confidence
        return max(0.0, min(1.0, score))

    def assign_broad_topics(self, paper_items, candidates):
        if not paper_items or not candidates:
            return {}

        lines = [
            "Assign one broad research area to each paper.",
            "Choose only from the candidate ids below.",
            "Broad areas are disciplines or large fields, not detailed keywords.",
            'Return JSON only: {"assignments":{"paper_id":"broad_topic_001"}}',
            "",
            "Candidates:",
        ]
        for candidate in candidates:
            variants = ", ".join(candidate.get("variants", [])[1:8])
            detail = f" ({variants})" if variants else ""
            lines.append(f"{candidate['id']}: {candidate['canonical']}{detail}")
        lines.append("")
        lines.append("Papers:")
        for item in paper_items:
            lines.append(f"{item['paper_id']}: {item['text']}")

        response = self._chat("\n".join(lines))
        parsed = self._extract_json(response)
        assignments = parsed.get("assignments", {}) if parsed else {}
        valid_ids = {candidate["id"] for candidate in candidates}
        return {
            paper_id: topic_id
            for paper_id, topic_id in assignments.items()
            if topic_id in valid_ids
        }
