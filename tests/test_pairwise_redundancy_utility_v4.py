import unittest

from hela_mem.pairwise_redundancy_utility_v4 import (
    build_gap_match_prompt, build_pair_prompt, parse_gap_match_response,
    parse_pair_response, redundancy_diagnostics, replay_eligibility,
)


class PairwiseRedundancyUtilityV4Test(unittest.TestCase):
    def test_pair_prompt_is_gold_free_and_pairwise(self):
        prompt = build_pair_prompt("Where did they meet?", "They met at a conference.", "They first met at a conference.")
        self.assertIn("SAME_FACT", prompt)
        self.assertNotIn("REFERENCE ANSWER", prompt.upper())

    def test_label_parsers_accept_only_first_line_labels(self):
        self.assertEqual(parse_pair_response("SAME_FACT\nreason"), "SAME_FACT")
        self.assertEqual(parse_gap_match_response("FILLS_GAP"), "FILLS_GAP")
        with self.assertRaises(ValueError): parse_pair_response("They are SAME_FACT")

    def test_gap_prompt_excludes_duplicate_label(self):
        prompt = build_gap_match_prompt("Q", {"base_coverage": [], "answerable_from_base": False, "missing_slots": [{"slot_id": "G1", "description": "place"}]}, "C")
        self.assertIn("FILLS_GAP", prompt)
        self.assertNotIn("DUPLICATE", prompt)

    def test_diagnostics_include_requested_transitions(self):
        rows = [{"question_id": "q", "candidate_id": "c", "candidate_text": "c", "matched_base_memory_id": "1", "matched_base_memory_text": "b", "oracle_label": "IRRELEVANT", "predicted_label": "REDUNDANT"}]
        result = redundancy_diagnostics(rows)
        self.assertEqual(result["summary"]["oracle_irrelevant_to_predicted_redundant"], 1)
        self.assertEqual(len(result["records"]), 1)

    def test_replay_gate_requires_all_targets(self):
        good = {"per_class": {"SUPPORTING": {"precision": .31}, "REDUNDANT": {"recall": .21}, "IRRELEVANT": {"support": 100}}, "irrelevant_to_supporting": 5}
        bad = {"per_class": {"SUPPORTING": {"precision": .31}, "REDUNDANT": {"recall": .19}, "IRRELEVANT": {"support": 100}}, "irrelevant_to_supporting": 5}
        self.assertTrue(replay_eligibility(good)["run_retrieval_replay"])
        self.assertFalse(replay_eligibility(bad)["run_retrieval_replay"])


if __name__ == "__main__":
    unittest.main()
