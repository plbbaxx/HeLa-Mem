import unittest

import numpy as np

from hela_mem.scan_oracle_edge_utility import edge_multiplier, select_flipped, spread_scores, summarize


class OracleEdgeUtilityScanTest(unittest.TestCase):
    def test_label_multipliers_apply_only_to_cross_edges(self):
        base = {"0", "1"}
        labels = {"2": "supporting", "3": "irrelevant"}
        self.assertEqual(edge_multiplier("0", "2", base, labels, 1.0, 0.2), (2.0, "supporting"))
        self.assertEqual(edge_multiplier("0", "3", base, labels, 1.0, 0.2), (0.0, "irrelevant"))
        self.assertEqual(edge_multiplier("0", "1", base, labels, 1.0, 0.2), (1.0, None))
        self.assertEqual(edge_multiplier("2", "0", base, labels, 1.0, 0.2), (1.0, None))

    def test_oracle_changes_only_calibrated_propagation(self):
        node_ids = ["0", "1", "2"]
        activations = np.array([0.9, 0.4, 0.4])
        edges = {"0": {"1": 0.5, "2": 0.5}}
        original = spread_scores(activations, node_ids, edges, 0.1, 0.4)
        oracle = spread_scores(activations, node_ids, edges, 0.1, 0.4, {"0"}, {"1": "supporting", "2": "irrelevant"}, 1.0, 0.2)
        self.assertAlmostEqual(original[1], 0.445)
        self.assertAlmostEqual(original[2], 0.445)
        self.assertAlmostEqual(oracle[1], 0.49)
        self.assertAlmostEqual(oracle[2], 0.4)

    def test_fixed_base_selection(self):
        scores = np.array([0.8, 0.9, 0.7, 0.85])
        self.assertEqual(select_flipped(scores, ["0", "1", "2", "3"], ["0", "1"], 3, 2), ["3"])

    def test_success_contract_is_predeclared(self):
        records = []
        for index in range(10):
            records.append({"selection_changed": True, "supporting_added_ids": [str(index)], "non_supporting_added_ids": [], "irrelevant_removed_ids": [], "zero_to_supporting": False, "added_labels": {str(index): "supporting"}, "removed_labels": {}})
        report = summarize(records, 10, 0.5)
        self.assertTrue(report["success_contract"]["overall_pass"])


if __name__ == "__main__":
    unittest.main()
