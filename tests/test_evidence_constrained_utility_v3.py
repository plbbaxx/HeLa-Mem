import unittest

from hela_mem.evidence_constrained_utility_v3 import (
    build_match_prompt,
    parse_stage2_response,
    redundant_diagnostics,
    replay_eligibility,
)


class EvidenceConstrainedUtilityV3Test(unittest.TestCase):
    def test_prompt_contains_ids_and_no_gold_interface(self):
        prompt = build_match_prompt(
            "Where did they meet?",
            [{"memory_id": "42", "content": "Sarah influenced the career."}],
            {"missing_slots": [{"slot_id": "G1", "description": "meeting location"}]},
            "They met at a conference.",
        )
        self.assertIn("Base Memory ID: 42", prompt)
        self.assertIn("meeting location", prompt)
        self.assertNotIn("REFERENCE ANSWER", prompt.upper())

    def test_stage2_requires_evidence_for_each_label(self):
        self.assertEqual(parse_stage2_response('{"label":"IRRELEVANT"}')["label"], "IRRELEVANT")
        self.assertEqual(parse_stage2_response('{"label":"REDUNDANT","base_memory_id":"7","duplicated_fact":"Sarah influenced the career"}')["base_memory_id"], "7")
        with self.assertRaises(ValueError):
            parse_stage2_response('{"label":"REDUNDANT"}')
        with self.assertRaises(ValueError):
            parse_stage2_response('{"label":"IRRELEVANT","evidence":"extra"}')

    def test_redundant_diagnostics_check_id_and_fact_grounding(self):
        rows = [{
            "predicted_label": "REDUNDANT", "question_id": "q", "candidate_id": "c",
            "candidate": "candidate", "oracle_label": "IRRELEVANT", "base_memory_id_exists": True,
            "referenced_base_memory": "Sarah influenced the user's career choice.",
            "decision": {"base_memory_id": "1", "duplicated_fact": "Sarah influenced career choice"},
        }]
        records, summary = redundant_diagnostics(rows)
        self.assertTrue(records[0]["duplicated_fact_lexical_grounding_proxy"])
        self.assertEqual(summary["false_redundant_count"], 1)

    def test_replay_is_gated_by_redundant_recall_and_safety(self):
        v2 = {"classification_metrics": {"per_class": {"SUPPORTING": {"precision": 0.38}}}}
        good = {"per_class": {"SUPPORTING": {"precision": 0.37}, "REDUNDANT": {"recall": 0.21}, "IRRELEVANT": {"support": 100}}, "irrelevant_to_supporting": 5}
        bad = {"per_class": {"SUPPORTING": {"precision": 0.37}, "REDUNDANT": {"recall": 0.19}, "IRRELEVANT": {"support": 100}}, "irrelevant_to_supporting": 5}
        self.assertTrue(replay_eligibility(good, v2)["run_retrieval_replay"])
        self.assertFalse(replay_eligibility(bad, v2)["run_retrieval_replay"])


if __name__ == "__main__":
    unittest.main()
