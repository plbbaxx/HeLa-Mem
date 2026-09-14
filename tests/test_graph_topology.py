import unittest

from hela_mem.analyze_graph_topology import aggregate, category, undirected_edges


class GraphTopologyTest(unittest.TestCase):
    def test_undirected_edges_deduplicate_bidirectional_storage(self):
        edges = {
            "0": {"1": 0.5, "2": 0.2},
            "1": {"0": 0.7},
            "2": {"0": 0.2},
        }
        result = list(undirected_edges(edges))
        self.assertEqual(result, [("0", "1", 0.7), ("0", "2", 0.2)])

    def test_categories_and_aggregate(self):
        self.assertEqual(category("0", "1", {"0", "1"}), "base_base")
        self.assertEqual(category("0", "2", {"0", "1"}), "base_nonbase")
        self.assertEqual(category("2", "3", {"0", "1"}), "nonbase_nonbase")
        summary = aggregate([{
            "missing_graph": False,
            "has_base_nonbase_edge": True,
            "edge_counts": {"base_base": 1, "base_nonbase": 2, "nonbase_nonbase": 1},
            "edge_weight_mass": {"base_base": 1.0, "base_nonbase": 2.0, "nonbase_nonbase": 1.0},
            "total_undirected_edges": 4,
        }])
        self.assertEqual(summary["edges_by_category"]["base_nonbase"]["edge_count"], 2)
        self.assertEqual(summary["edges_by_category"]["base_nonbase"]["edge_count_share"], 0.5)
        self.assertEqual(summary["questions_with_base_nonbase_edge"], 1)


if __name__ == "__main__":
    unittest.main()
