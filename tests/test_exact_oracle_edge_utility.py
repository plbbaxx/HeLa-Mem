import unittest

from hela_mem.scan_exact_oracle_edge_utility import (
    labels_for_current_base,
    multipliers_for_record,
    summarize,
)


class ExactOracleEdgeUtilityTest(unittest.TestCase):
    def test_reuses_labels_only_for_identical_base_and_complete_targets(self):
        current = {"base_top_k_ids": ["0"], "candidate_targets": [{"candidate_id": "2"}]}
        cached = {"base_top_k_ids": ["0"], "candidate_annotations": [{"candidate_id": "2", "label": "supporting"}]}
        labels, source = labels_for_current_base(current, cached, "unused", 4, 1)
        self.assertEqual(source, "reused_quality_report")
        self.assertEqual(labels["2"]["label"], "supporting")

    def test_builds_only_directed_base_to_candidate_multipliers(self):
        record = {"candidate_targets": [{"candidate_id": "2", "connected_base_ids": ["0", "1"]}]}
        values = multipliers_for_record(record, {"2": {"label": "redundant"}})
        self.assertEqual(values, {("0", "2"): 0.2, ("1", "2"): 0.2})

    def test_requested_four_metrics(self):
        records = [{
            "selection_changed": True,
            "supporting_added_ids": ["2"],
            "irrelevant_removed_ids": ["3"],
            "rescued_useful_question": True,
            "saved_current_base_exact_match": True,
            "label_source": "reused_quality_report",
            "keyword_status": "ok",
        }]
        report = summarize(records)
        self.assertEqual(report["valid_retrieval_questions"], 1)
        self.assertEqual(report["flipped_selection_changed_questions"], 1)
        self.assertEqual(report["supporting_candidates_added"], 1)
        self.assertEqual(report["irrelevant_candidates_removed"], 1)
        self.assertEqual(report["rescued_useful_questions"], 1)


if __name__ == "__main__":
    unittest.main()
