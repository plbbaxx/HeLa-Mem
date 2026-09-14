import unittest

from hela_mem.structured_residual_utility_v2 import (
    build_gap_prompt,
    build_match_prompt,
    compare_with_v1,
    evaluation_gate,
    parse_gap_response,
    parse_stage2_response,
)


class StructuredResidualUtilityV2Test(unittest.TestCase):
    def test_prompts_have_no_gold_answer_interface(self):
        residual = {"base_coverage": ["career influence"], "answerable_from_base": False, "missing_slots": [{"slot_id": "G1", "description": "meeting location"}]}
        gap = build_gap_prompt("Where did they meet?", [{"content": "Sarah influenced a career."}])
        match = build_match_prompt("Where did they meet?", residual, "They met at a conference.")
        self.assertNotIn("REFERENCE ANSWER", gap.upper())
        self.assertNotIn("REFERENCE ANSWER", match.upper())
        self.assertIn("meeting location", match)

    def test_gap_response_is_structured_and_consistent(self):
        parsed = parse_gap_response('{"base_coverage":["career influence"],"answerable_from_base":false,"missing_slots":[{"slot_id":"G1","description":"meeting location"}]}')
        self.assertEqual(parsed["missing_slots"][0]["slot_id"], "G1")
        with self.assertRaises(ValueError):
            parse_gap_response('{"base_coverage":[],"answerable_from_base":true,"missing_slots":[{"description":"location"}]}')

    def test_stage2_output_is_constrained(self):
        self.assertEqual(parse_stage2_response("fills_gap"), "FILLS_GAP")
        self.assertEqual(parse_stage2_response("FILLS_GAP\n\nExplanation: candidate supplies the location."), "FILLS_GAP")
        with self.assertRaises(ValueError):
            parse_stage2_response("The label is FILLS_GAP")

    def test_predeclared_gate(self):
        classification = {
            "per_class": {
                "SUPPORTING": {"precision": 0.31, "recall": 0.41, "support": 37},
                "REDUNDANT": {"precision": 0.3, "recall": 0.21, "support": 43},
                "IRRELEVANT": {"precision": 0.8, "recall": 0.7, "support": 261},
            },
            "irrelevant_to_supporting": 80,
        }
        replay = {"oracle_selection_precision_on_changed_questions": 0.36, "useful_questions_rescued": 1}
        self.assertTrue(evaluation_gate(classification, replay)["recommended_for_final_answer_evaluation"])

    def test_v1_comparison_reports_directional_deltas(self):
        v1 = {
            "classification_metrics": {
                "macro_f1": 0.2, "irrelevant_to_supporting": 20,
                "per_class": {"SUPPORTING": {"precision": 0.1, "recall": 0.8}, "REDUNDANT": {"recall": 0.0}, "IRRELEVANT": {"support": 100}},
            },
            "retrieval_replay_metrics": {"oracle_selection_precision": 0.1, "useful_questions_rescued": 2},
        }
        v2_classification = {
            "macro_f1": 0.4, "irrelevant_to_supporting": 10,
            "per_class": {"SUPPORTING": {"precision": 0.3, "recall": 0.5}, "REDUNDANT": {"recall": 0.2}, "IRRELEVANT": {"support": 100}},
        }
        replay = {"oracle_selection_precision_on_changed_questions": 0.4, "useful_questions_rescued": 3}
        result = compare_with_v1(v1, v2_classification, replay)
        self.assertAlmostEqual(result["supporting_precision_delta"], 0.2)
        self.assertAlmostEqual(result["irrelevant_to_supporting_rate_delta"], -0.1)


if __name__ == "__main__":
    unittest.main()
