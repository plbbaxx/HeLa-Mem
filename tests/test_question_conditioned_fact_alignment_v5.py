import unittest
from hela_mem.question_conditioned_fact_alignment_v5 import build_alignment_prompt, build_base_claim_prompt, build_candidate_claim_prompt, error_exports, parse_alignment, parse_base_claims, parse_candidate_claim

class V5Test(unittest.TestCase):
    def test_claim_parsers_are_strict(self):
        claims=parse_base_claims('{"claims":[{"memory_id":"1","answer_relevant":true,"slot":"place","value":"conference"},{"memory_id":"2","answer_relevant":false,"slot":null,"value":null}]}',["1","2"])
        self.assertEqual(len(claims),2); self.assertFalse(parse_candidate_claim('{"answer_relevant":false,"slot":null,"value":null}')["answer_relevant"])
        self.assertFalse(parse_base_claims('{"claims":[]}',["1"])[0]["answer_relevant"])
        recovered=parse_base_claims('{"claims":[{"memory_id":"9","answer_relevant":false,"slot":null,"value":null},{"memory_id":"1","answer_relevant":true,"slot":"place","value":"conference"},{"memory_id":"1","answer_relevant":true,"slot":"date","value":"Monday"}]}',["1"])
        self.assertEqual(recovered[0]["value"],"conference")
        self.assertFalse(parse_candidate_claim('{"answer_relevant":true,"slot":"place"}')["answer_relevant"])
    def test_alignment_requires_valid_base_for_redundancy(self):
        parsed=parse_alignment('{"label":"REDUNDANT","matched_base_memory_id":"1","matched_slot":"place","reason":"same place"}',["1"])
        self.assertEqual(parsed["label"],"REDUNDANT")
        self.assertEqual(parse_alignment('{"label":"REDUNDANT","matched_base_memory_id":"2","matched_slot":"place"}',["1"])["label"],"IRRELEVANT")
        self.assertEqual(parse_alignment('{"label":"supporting"}',["1"])["label"],"SUPPORTING")
    def test_prompts_are_gold_free(self):
        self.assertNotIn("REFERENCE ANSWER",build_base_claim_prompt("Q",[{"memory_id":"1","content":"x"}]).upper())
        self.assertIn("CANDIDATE MEMORY",build_candidate_claim_prompt("Q","x"))
        self.assertIn("MISSING INFORMATION",build_alignment_prompt("Q",{"missing_slots":[]},[],{"answer_relevant":False,"slot":None,"value":None}))
    def test_error_export(self):
        row={"oracle_label":"SUPPORTING","predicted_label":"REDUNDANT","candidate_answer_relevant":True,"candidate_slot":"place","candidate_value":"conference","matched_base_slot":"place","matched_base_value":"office","question_type":"single-session-user"}
        self.assertEqual(len(error_exports([row])["oracle_supporting_to_predicted_redundant"]),1)
if __name__=="__main__": unittest.main()
