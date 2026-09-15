import json
import tempfile
import unittest
from pathlib import Path

from hela_mem.analyze_ppr_associative_expansion import analyze
from hela_mem.ppr_expansion import (
    personalized_pagerank,
    ppr_associative_candidates,
    strongest_shortest_paths,
)


class PPRExpansionTest(unittest.TestCase):
    def setUp(self):
        self.node_ids = ["a", "b", "c", "d"]
        self.edges = {
            "a": {"b": 1.0},
            "b": {"a": 0.2, "c": 0.8},
            "c": {"b": 1.0},
            "d": {},
        }

    def test_ppr_is_normalized_and_does_not_reach_disconnected_nodes(self):
        result = personalized_pagerank(
            self.node_ids, self.edges, {"a": 1.0}, damping=0.5, max_iter=100, tol=1e-10
        )
        self.assertAlmostEqual(sum(result["scores"].values()), 1.0)
        self.assertEqual(result["scores"]["d"], 0.0)
        self.assertTrue(result["converged"])

    def test_candidates_exclude_base_and_include_multihop_path(self):
        result = ppr_associative_candidates(
            self.node_ids, self.edges, ["a"], {"a": 0.7}, top_n=3, damping=0.5
        )
        by_id = {row["node_id"]: row for row in result["candidates"]}
        self.assertNotIn("a", by_id)
        self.assertEqual(by_id["b"]["hop_distance"], 1)
        self.assertEqual(by_id["c"]["hop_distance"], 2)
        self.assertEqual(by_id["c"]["hebbian_path"], ["a", "b", "c"])
        self.assertNotIn("d", by_id)

    def test_strongest_predecessor_is_selected_within_shortest_paths(self):
        edges = {
            "a": {"b": 0.4, "c": 0.8},
            "b": {"d": 0.9},
            "c": {"d": 0.2},
            "d": {},
        }
        paths = strongest_shortest_paths(["a", "b", "c", "d"], edges, ["a"])
        self.assertEqual(paths["d"]["hop_distance"], 2)
        self.assertEqual(paths["d"]["strongest_predecessor"], "b")
        self.assertEqual(paths["d"]["hebbian_path"], ["a", "b", "d"])

    def test_invalid_damping_is_rejected(self):
        with self.assertRaises(ValueError):
            personalized_pagerank(self.node_ids, self.edges, {"a": 1.0}, damping=1.0)

    def test_offline_replay_reports_multihop_supporting_addition(self):
        dataset = [{"question_id": "q1", "question_type": "multi-session"}]
        predictions = {"q1": {"retrieved_episodic": [
            {"node_id": "a", "source": "base", "base_score": 0.9},
        ]}}
        graph = {
            "nodes": {node_id: {"id": node_id, "content": node_id} for node_id in self.node_ids},
            "edges": self.edges,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "q1_hebbian.json").write_text(json.dumps(graph), encoding="utf-8")
            report = analyze(
                dataset, predictions, root, top_k=1, damping=0.5, ppr_top_n=2,
                one_hop_seed_k=1, one_hop_neighbor_k=1,
                quality={"q1": {"b": "irrelevant", "c": "supporting"}},
            )
        record = report["records"][0]
        self.assertEqual(record["one_hop_candidate_ids"], ["b"])
        self.assertEqual(record["ppr_candidate_ids"], ["b", "c"])
        self.assertEqual(report["quality"]["supporting_added"], 1)
        self.assertEqual(report["quality"]["supporting_recall"], 1.0)


if __name__ == "__main__":
    unittest.main()
