import unittest

from hela_mem.build_utility_dataset import build_pairs, describe, gold_logprob_values, graph_candidates, stratified_split, transition


class UtilityDatasetTest(unittest.TestCase):
    def test_candidates_are_unique_nonbase_neighbors(self):
        item = {"question_id": "q"}
        prediction = {"retrieved_episodic": [{"source": "base", "node_id": "a"}]}
        graph = {"nodes": {"a": {}, "b": {}, "c": {}}, "edges": {"a": {"b": .5, "c": .2}}}
        base, candidates = graph_candidates(item, prediction, graph)
        self.assertEqual(base, ["a"])
        self.assertEqual([row["candidate_memory_id"] for row in candidates], ["b", "c"])

    def test_pairs_never_cross_questions(self):
        rows = [
            {"question_id": "q", "candidate_memory_id": "a", "utility_score": .2, "semantic_score": .1},
            {"question_id": "q", "candidate_memory_id": "b", "utility_score": 0, "semantic_score": .2},
            {"question_id": "x", "candidate_memory_id": "c", "utility_score": -1, "semantic_score": .3},
        ]
        pairs = build_pairs(rows, .02, .15, .5, .1)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["question_id"], "q")

    def test_stats_split_transition(self):
        self.assertEqual(transition(False, True), "W2C")
        self.assertEqual(describe([1, 2])["median"], 1.5)

    def test_stratified_split_uses_only_candidate_questions(self):
        rows=[]
        for i in range(20):
            rows.extend([
                {"question_id":str(i),"utility_score":-.2 if i<6 else 0},
                {"question_id":str(i),"utility_score":.2 if i<9 else .01},
            ])
        splits,profiles=stratified_split(rows)
        self.assertEqual(sum(map(len,splits.values())),20)
        self.assertEqual(set().union(*map(set,splits.values())),set(profiles))
        self.assertTrue(all(any(profiles[q]["has_positive"] for q in splits[s]) for s in splits))
        self.assertTrue(all(any(profiles[q]["has_within_question_variation"] for q in splits[s]) for s in splits))

    def test_gold_logprobs_use_exact_gold_span(self):
        self.assertEqual(gold_logprob_values([None,-.4,-.3,-.2],2,4),[-.3,-.2])
        with self.assertRaisesRegex(RuntimeError,"missing gold token"):
            gold_logprob_values([None,-.4,None,-.2],2,4)
        with self.assertRaisesRegex(RuntimeError,"incomplete prompt"):
            gold_logprob_values([None,-.4,-.3],2,4)


if __name__ == "__main__":
    unittest.main()
