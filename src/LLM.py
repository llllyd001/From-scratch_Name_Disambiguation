import json
import os
import re
import time
import urllib.error
import urllib.request


DEFAULT_MODEL_NAME = "deepseek-chat"
DEFAULT_BASE_URL = "https://models.sjtu.edu.cn/api/v1"


class OneShotLLMClusterer:
    def __init__(
        self,
        model_name=DEFAULT_MODEL_NAME,
        max_new_tokens=512,
        max_retries=5,
        retry_delay=30,
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.base_url = os.environ.get("ZHIYUAN_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
        self.api_key = os.environ.get("ZHIYUAN_API_KEY", "")

    def _shorten(self, text, limit):
        text = " ".join((text or "").split())
        return text[:limit]

    def _compact_name(self, name):
        return re.sub(r"[^a-z]", "", name.lower())

    def _is_target_author(self, author_name, target_name):
        parts = target_name.replace("_", " ").split()
        variants = {"".join(parts), "".join(reversed(parts))}
        return self._compact_name(author_name) in variants

    def _target_org(self, paper, target_name):
        for author in paper.get("authors", []):
            if self._is_target_author(author.get("name", ""), target_name):
                return author.get("org", "")
        return ""

    def _org_text(self, paper, target_name):
        return self._shorten(self._target_org(paper, target_name), 260)

    def _make_org_items(self, papers, paper_ids, target_name):
        org_to_papers = {}
        for paper_id in paper_ids:
            key = self._org_text(papers[paper_id], target_name)
            org_to_papers.setdefault(key, []).append(paper_id)

        items = []
        for index, (org, ids) in enumerate(org_to_papers.items(), start=1):
            alias = f"o{index}"
            items.append({"alias": alias, "org": org, "paper_ids": ids})
        return items

    def _org_line(self, item):
        return " | ".join(
            [
                item["alias"],
                f"org={item['org']}",
                f"papers={len(item['paper_ids'])}",
            ]
        )

    def _build_prompt(self, org_items, target_name):
        lines = [
            f"Cluster organization records for the ambiguous author name: {target_name}.",
            "Use only the target author's organization evidence.",
            "Treat equivalent organization spellings as the same evidence.",
            "If organization is empty or uncertain, omit that record.",
            'Return JSON only: {"clusters":[["o1","o2"],["o5","o8"]]}.',
            "Rules:",
            "- use only aliases o1, o2, ... shown below",
            "- only output clusters with at least two organization records",
            "- omit uncertain records; they will become singleton clusters automatically",
            "- no explanation, no markdown, no extra keys",
            "",
            "Organization records:",
        ]
        for item in org_items:
            lines.append(self._org_line(item))
        return "\n".join(lines)

    def _chat(self, prompt):
        if not self.api_key:
            raise RuntimeError("ZHIYUAN_API_KEY is not set")

        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": self.max_new_tokens,
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
        if start == -1:
            return None

        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start:index + 1]
        return None

    def _normalize_alias(self, value, aliases):
        if isinstance(value, int):
            if 1 <= value <= len(aliases):
                return aliases[value - 1]
            if 0 <= value < len(aliases):
                return aliases[value]
            return None

        if not isinstance(value, str):
            return None

        value = value.strip()
        alias_match = re.fullmatch(r"(?:org|o)[_\-\s]*(\d+)", value, flags=re.IGNORECASE)
        if alias_match:
            index = int(alias_match.group(1))
            if 1 <= index <= len(aliases):
                return aliases[index - 1]
        if value in aliases:
            return value
        return None

    def _normalize_clusters(self, parsed, aliases):
        clusters = parsed.get("clusters") if isinstance(parsed, dict) else parsed
        if not isinstance(clusters, list):
            return None

        normalized = []
        seen = set()
        for cluster in clusters:
            if not isinstance(cluster, list):
                continue
            cleaned = []
            for item in cluster:
                alias = self._normalize_alias(item, aliases)
                if alias is None or alias in seen:
                    continue
                cleaned.append(alias)
                seen.add(alias)
            if len(cleaned) >= 2:
                normalized.append(cleaned)

        for alias in aliases:
            if alias not in seen:
                normalized.append([alias])

        return sorted(normalized, key=lambda ids: (-len(ids), ids[0]))

    def _fallback_clusters(self, paper_ids, generated_text, reason):
        preview = self._shorten(generated_text, 300)
        print(f"LLM output ignored: {reason}", flush=True)
        if preview:
            print(f"LLM output preview: {preview}", flush=True)
        return [[paper_id] for paper_id in paper_ids]

    def _expand_org_clusters(self, org_clusters, org_items):
        item_by_alias = {item["alias"]: item for item in org_items}
        paper_clusters = []
        for cluster in org_clusters:
            paper_ids = []
            for alias in cluster:
                paper_ids.extend(item_by_alias[alias]["paper_ids"])
            paper_clusters.append(paper_ids)
        return sorted(paper_clusters, key=lambda ids: (-len(ids), ids[0]))

    def cluster_target(self, papers, paper_ids, target_name):
        org_items = self._make_org_items(papers, paper_ids, target_name)
        aliases = [item["alias"] for item in org_items]
        prompt = self._build_prompt(org_items, target_name)
        generated_text = self._chat(prompt)

        parsed = None
        chunk = self._extract_json(generated_text)
        if chunk is not None:
            try:
                parsed = json.loads(chunk)
            except json.JSONDecodeError:
                parsed = None

        if parsed is None:
            return self._fallback_clusters(paper_ids, generated_text, "no valid JSON found")

        org_clusters = self._normalize_clusters(parsed, aliases)
        if org_clusters is None:
            return self._fallback_clusters(
                paper_ids,
                generated_text,
                "JSON did not contain a usable clusters list",
            )
        return self._expand_org_clusters(org_clusters, org_items)
