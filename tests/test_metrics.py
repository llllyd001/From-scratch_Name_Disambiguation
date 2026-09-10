import unittest

from name_disambiguation.core.metrics import b3_metrics, pairwise_metrics


IDS = ["p1", "p2", "p3", "p4"]
TRUTH = [{"p1", "p2"}, {"p3", "p4"}]


class MetricsTest(unittest.TestCase):
    def test_perfect_clustering_scores_one(self):
        for metric in (pairwise_metrics, b3_metrics):
            self.assertEqual(
                metric(TRUTH, TRUTH, IDS),
                {"precision": 1.0, "recall": 1.0, "f1": 1.0},
            )

    def test_all_singletons_have_zero_pairwise_recall(self):
        predicted = [{paper_id} for paper_id in IDS]
        self.assertEqual(
            pairwise_metrics(TRUTH, predicted, IDS),
            {"precision": 0, "recall": 0.0, "f1": 0},
        )

    def test_all_merged_b3_values(self):
        result = b3_metrics(TRUTH, [set(IDS)], IDS)
        self.assertAlmostEqual(result["precision"], 0.5)
        self.assertAlmostEqual(result["recall"], 1.0)
        self.assertAlmostEqual(result["f1"], 2 / 3)


if __name__ == "__main__":
    unittest.main()
