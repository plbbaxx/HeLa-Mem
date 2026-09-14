import unittest

from hela_mem.utility_scorer_v1 import (
    CLASSES,
    build_scorer_prompt,
    classification_metrics,
    parse_label,
    replay_metrics,
)


class UtilityScorerV1Test(unittest.TestCase):
    def test_prompt_has_no_gold_answer_interface_or_leakage(self):
        sentinel = "SECRET_GOLD_ANSWER"
        prompt = build_scorer_prompt(
            "Where did they meet?",
            [{"content": "They discussed career planning."}],
            "They met at a conference.",
        )
        self.assertNotIn(sentinel, prompt)
        self.assertNotIn("REFERENCE ANSWER", prompt.upper())
        self.assertIn("FULL Base Memories", prompt)

    def test_output_must_be_exactly_one_label(self):
        for label in CLASSES:
            self.assertEqual(parse_label(label.lower()), label)
        with self.assertRaises(ValueError):
            parse_label("SUPPORTING because it helps")

    def test_classification_metrics_and_requested_errors(self):
        rows = [
            {"oracle_label": "SUPPORTING", "predicted_label": "SUPPORTING"},
            {"oracle_label": "SUPPORTING", "predicted_label": "REDUNDANT"},
            {"oracle_label": "IRRELEVANT", "predicted_label": "SUPPORTING"},
            {"oracle_label": "REDUNDANT", "predicted_label": "IRRELEVANT"},
        ]
        metrics = classification_metrics(rows)
        self.assertEqual(metrics["false_supporting_count"], 1)
        self.assertEqual(metrics["supporting_to_redundant"], 1)
        self.assertEqual(metrics["supporting_to_irrelevant"], 0)
        self.assertEqual(metrics["irrelevant_to_supporting"], 1)
        self.assertAlmostEqual(metrics["overall_accuracy"], 0.25)

    def test_oracle_overlap_metrics(self):
        rows = [{
            "selection_changed_from_baseline": True,
            "supporting_added_ids": ["2"], "irrelevant_removed_ids": ["3"],
            "irrelevant_added_ids": [], "useful_question_rescued": True,
            "oracle_exact_match": False, "oracle_intersection": 1,
            "oracle_union": 2, "automatic_count": 1, "oracle_count": 2,
            "oracle_changed_from_baseline": True,
        }]
        metrics = replay_metrics(rows)
        self.assertEqual(metrics["selection_changed_questions"], 1)
        self.assertEqual(metrics["supporting_added"], 1)
        self.assertEqual(metrics["useful_questions_rescued"], 1)
        self.assertEqual(metrics["oracle_selection_recall"], 0.5)
        self.assertEqual(metrics["oracle_exact_match_rate_on_changed_questions"], 0.0)


if __name__ == "__main__":
    unittest.main()
