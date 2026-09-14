import unittest

from hela_mem.build_utility_dataset import build_pairs, describe, graph_candidates, split_ids, transition


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
        self.assertEqual(sum(map(len, split_ids([str(i) for i in range(20)]).values())), 20)
        self.assertEqual(describe([1, 2])["median"], 1.5)


if __name__ == "__main__":
    unittest.main()
