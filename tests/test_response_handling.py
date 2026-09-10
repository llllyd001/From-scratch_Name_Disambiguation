import json
import unittest

from name_disambiguation.core.llm_cluster import validate
from name_disambiguation.core.normalize_llm_response import normalize_response


def response_with(content):
    return {
        "choices": [{
            "finish_reason": "stop",
            "message": {"content": json.dumps(content)},
        }]
    }


class ResponseHandlingTest(unittest.TestCase):
    def test_compact_assignments_are_mapped_back_to_paper_ids(self):
        result = normalize_response(
            response_with({
                "assignments": {"0": "a", "1": "a", "2": "b"},
                "labels": {"a": "profile a", "b": "profile b"},
            }),
            paper_ids=["p1", "p2", "p3"],
        )
        self.assertEqual(result, {"clusters": [
            {"label": "profile a", "paper_ids": ["p1", "p2"]},
            {"label": "profile b", "paper_ids": ["p3"]},
        ]})

    def test_validate_deduplicates_ids_inside_one_cluster(self):
        result = {"clusters": [{"paper_ids": ["p1", "p1", "p2"]}]}
        self.assertEqual(validate(result, ["p1", "p2"]), [{"p1", "p2"}])
        self.assertEqual(result["clusters"][0]["paper_ids"], ["p1", "p2"])

    def test_validate_rejects_cross_cluster_duplicates(self):
        result = {"clusters": [
            {"paper_ids": ["p1", "p2"]},
            {"paper_ids": ["p2", "p3"]},
        ]}
        with self.assertRaisesRegex(ValueError, "multiple clusters"):
            validate(result, ["p1", "p2", "p3"])

    def test_validate_rejects_missing_ids(self):
        with self.assertRaisesRegex(ValueError, "missing 1 paper IDs"):
            validate({"clusters": [{"paper_ids": ["p1"]}]}, ["p1", "p2"])


if __name__ == "__main__":
    unittest.main()
