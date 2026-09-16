import copy
import json
import sys
import types
import unittest

import numpy as np

try:
    openai_module = __import__("openai")
except ImportError:
    openai_module = None
if openai_module is None or not hasattr(openai_module, "OpenAI"):
    module = types.ModuleType("openai")
    module.OpenAI = object
    sys.modules["openai"] = module

from hela_mem.raw_actr_retriever import RawACTRRetriever, _parse_cues, summarize_fans


class NodesOnlyGraph:
    def __init__(self):
        self.nodes = {
            "m1": {"id": "m1", "content": "query-like", "embedding": [1.0, 0.0], "timestamp": "t"},
            "m2": {"id": "m2", "content": "relation answer", "embedding": [0.8, 0.6], "timestamp": "t"},
            "m3": {"id": "m3", "content": "same relation outside pool", "embedding": [0.0, 1.0], "timestamp": "t"},
            "m4": {"id": "m4", "content": "other", "embedding": [-1.0, 0.0], "timestamp": "t"},
        }

    @property
    def edges(self):
        raise AssertionError("RawACTRRetriever must never access graph edges")


class RawACTRRetrieverTest(unittest.TestCase):
    def setUp(self):
        self.graph = NodesOnlyGraph()
        self.vectors = {
            "question": np.array([1.0, 0.0]),
            "relation cue": np.array([0.0, 1.0]),
            "broad cue": np.array([1.0, 1.0]),
        }
        self.extractor = lambda query: [
            {"type": "relation", "text": "relation cue"},
            {"type": "event", "text": "broad cue"},
        ]
        self.embedding = lambda text: self.vectors[text]

    def retriever(self):
        return RawACTRRetriever(self.graph, self.extractor, self.embedding)

    def test_raw_and_actr_share_coarse_pool_and_nodes_are_unchanged(self):
        before = copy.deepcopy(self.graph.nodes)
        raw = self.retriever()
        raw.retrieve("question", mode="raw", candidate_k=2, top_k=2)
        actr = self.retriever()
        actr.retrieve("question", mode="actr", candidate_k=2, top_k=2, fan_threshold=0.5)
        self.assertEqual(raw.last_diagnostic["coarse_candidate_ids"], actr.last_diagnostic["coarse_candidate_ids"])
        self.assertEqual(before, self.graph.nodes)

    def test_fan_uses_complete_bank_and_actr_can_change_ranking(self):
        retriever = self.retriever()
        results = retriever.retrieve(
            "question", mode="actr", candidate_k=2, top_k=2,
            fan_threshold=0.5, score_mode="actr",
        )
        diagnostic = retriever.last_diagnostic
        self.assertEqual(diagnostic["fan_computed_over_memory_count"], 4)
        self.assertEqual(diagnostic["coarse_candidate_ids"], ["m1", "m2"])
        self.assertEqual(results[0]["node"]["id"], "m2")
        self.assertNotEqual(
            diagnostic["coarse_candidate_ids"],
            [row["node"]["id"] for row in results],
        )

    def test_cue_idf_formula_and_finite_source_activation(self):
        retriever = self.retriever()
        retriever.retrieve("question", mode="cue_idf", candidate_k=2, top_k=2, fan_threshold=0.5)
        diagnostic = retriever.last_diagnostic
        self.assertTrue(all(abs(cue["source_activation"] - 0.5) < 1e-9 for cue in diagnostic["cues"]))
        candidate = diagnostic["candidates"][0]
        expected = sum(
            match["similarity"] * diagnostic["cues"][index]["fan_strength"]
            for index, match in enumerate(candidate["cue_matches"])
        )
        self.assertAlmostEqual(candidate["cue_idf_score"], expected)

    def test_parse_json_prompt_output_and_fan_summary(self):
        parsed = _parse_cues(json.dumps({"cues": [{"type": "event", "text": "Alice left Boston"}]}), "q")
        self.assertEqual(parsed[0]["text"], "Alice left Boston")
        summary = summarize_fans([{
            "memory_bank_size": 10,
            "cues": [{"fan": 1}, {"fan": 7}],
        }])
        self.assertEqual(summary["fan_histogram"], {"1": 1, "7": 1})
        self.assertEqual(summary["high_fan_cue_rate"], 0.5)
        self.assertEqual(summary["low_fan_cue_rate"], 0.5)

    def test_isolated_entity_cues_are_rejected(self):
        with self.assertRaises(ValueError):
            _parse_cues(json.dumps({"cues": [
                {"type": "relation", "text": "Alice"},
                {"type": "time", "text": "2024"},
            ]}), "Where did Alice move in 2024?")


if __name__ == "__main__":
    unittest.main()
