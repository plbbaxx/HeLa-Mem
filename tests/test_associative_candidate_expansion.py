import unittest

from hela_mem.analyze_associative_candidate_expansion import (
    expand_threshold_neighbors,
    expand_top_neighbors,
    summarize,
)


class AssociativeCandidateExpansionTest(unittest.TestCase):
    def setUp(self):
        self.edges = {
            "a": {"b": 0.9, "c": 0.8, "d": 0.1},
            "b": {"c": 0.7, "e": 0.6},
        }

    def test_top_neighbors_uses_base_order_and_excludes_base(self):
        seeds, candidates = expand_top_neighbors(["a", "b"], self.edges, 2, 2)
        self.assertEqual(seeds, ["a", "b"])
        self.assertEqual(candidates, ["c", "e"])

    def test_threshold_neighbors_preserves_only_strong_non_base_edges(self):
        seeds, candidates = expand_threshold_neighbors(["a", "b"], self.edges, 2, 0.7)
        self.assertEqual(seeds, ["a", "b"])
        self.assertEqual(candidates, ["c"])

    def test_summary_bins(self):
        result = summarize([{"candidate_count": 0}, {"candidate_count": 1}, {"candidate_count": 2}])
        self.assertEqual(result["candidate_pool_zero"]["count"], 1)
        self.assertEqual(result["candidate_pool_one"]["count"], 1)
        self.assertEqual(result["candidate_pool_two_or_more"]["count"], 1)
        self.assertEqual(result["median_candidate_pool_size"], 1)


if __name__ == "__main__":
    unittest.main()
