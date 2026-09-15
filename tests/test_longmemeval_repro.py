import copy
import json
import sys
import tempfile
import types
from concurrent.futures import ThreadPoolExecutor
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

# The workstation may have no OpenAI package or the legacy package; production
# requires openai>=1, while these unit tests patch every external call.
try:
    openai_module = __import__("openai")
except ImportError:
    openai_module = None
if openai_module is None or not hasattr(openai_module, "OpenAI"):
    module = types.ModuleType("openai")
    module.OpenAI = object
    sys.modules["openai"] = module

from hela_mem import encode_longmemeval as enc
from hela_mem import eval_longmemeval as ev
from hela_mem.hebbian_memory import HebbianMemoryGraph, apply_lateral_inhibition


class FakeExtractionClient:
    def chat_completion(self, model=None, messages=None, temperature=0.0, max_tokens=2000):
        prompt = messages[-1]["content"]
        if "Assistant Knowledge Extraction" in prompt:
            return "【Assistant Knowledge】\n- None"
        if "Profile Merge Task" in prompt:
            return "1. Core Psychological Traits:\n- None"
        return "【User Profile】\n1. Core Psychological Traits:\n- None\n【User Data】\n- likes tea"


def fake_embedding(text, model_name=None):
    value = float((sum(ord(char) for char in text) % 17) + 1)
    return np.array([value, 1.0, value / 2.0], dtype=np.float32)


def item(index):
    return {
        "question_id": f"q{index}", "question": "What does the user like?",
        "answer": "tea", "question_type": "single-session-user",
        "question_date": "2026-01-02",
        "haystack_dates": ["2026-01-01 00:00:00"],
        "haystack_sessions": [[
            {"role": "user", "content": "I like tea."},
            {"role": "assistant", "content": "I will remember that."},
        ]],
    }


class ReproPipelineTest(unittest.TestCase):
    def test_lateral_inhibition_formula(self):
        scores = np.array([1.0, 0.8, 0.2])
        actual = apply_lateral_inhibition(scores, beta=0.15, top_m=2)
        np.testing.assert_allclose(actual, [1.0, 0.77, 0.0])
        self.assertEqual(
            np.argsort(scores)[::-1].tolist(),
            np.argsort(actual)[::-1].tolist(),
        )

    def test_disabled_inhibition_is_baseline_equivalent(self):
        with tempfile.TemporaryDirectory() as temp:
            graph = HebbianMemoryGraph(str(Path(temp) / "memory.json"))
            graph.nodes = {
                str(i): {"id": str(i), "content": str(i), "embedding": [float(i), 1.0, 0.0], "timestamp": "2026-01-01", "keywords": []}
                for i in range(3)
            }
            with patch("hela_mem.hebbian_memory.get_embedding", lambda text: np.array([1.0, 1.0, 0.0])), \
                 patch("hela_mem.hebbian_memory.llm_extract_keywords", lambda text: set()), \
                 patch("hela_mem.hebbian_memory.compute_time_decay", lambda *args: 1.0):
                graph.use_inhibition = False
                graph.retrieve("query", top_k=2)
            trace = graph.last_retrieval_trace
            self.assertEqual(trace["final_scores"], trace["inhibited_scores"])
            self.assertEqual(trace["rank_before"], trace["rank_after"])
            self.assertEqual(trace["flipped_memory_ids_before"], trace["flipped_memory_ids_after"])

            graph.use_inhibition = True
            with patch("hela_mem.hebbian_memory.get_embedding", lambda text: np.array([1.0, 1.0, 0.0])), \
                 patch("hela_mem.hebbian_memory.llm_extract_keywords", lambda text: set()), \
                 patch("hela_mem.hebbian_memory.compute_time_decay", lambda *args: 1.0):
                graph.retrieve("query", top_k=2)
            inhibited_trace = graph.last_retrieval_trace
            self.assertEqual(trace["base_top_k_ids"], inhibited_trace["base_top_k_ids"])
            self.assertEqual(inhibited_trace["rank_before"], inhibited_trace["rank_after"])

    def test_retrieval_probe_can_disable_reinforcement(self):
        with tempfile.TemporaryDirectory() as temp:
            graph = HebbianMemoryGraph(str(Path(temp) / "memory.json"))
            graph.nodes = {
                "0": {"id": "0", "content": "a", "embedding": [1.0, 0.0], "timestamp": "2026-01-01", "keywords": []},
                "1": {"id": "1", "content": "b", "embedding": [0.0, 1.0], "timestamp": "2026-01-01", "keywords": []},
            }
            graph.add_edge("0", "1", weight=0.5, bidirectional=True)
            before = {source: dict(neighbors) for source, neighbors in graph.edges.items()}
            with patch("hela_mem.hebbian_memory.compute_time_decay", lambda *args: 1.0):
                graph.retrieve(
                    "query", top_k=1,
                    query_keywords_override=set(),
                    query_embedding_override=np.array([1.0, 0.0]),
                    current_time_override="2026-01-02 00:00:00",
                    edge_weight_multipliers={("0", "1"): 2.0},
                    update_graph=False,
                )
            self.assertEqual(before, {source: dict(neighbors) for source, neighbors in graph.edges.items()})
            self.assertFalse(graph.last_retrieval_trace["reinforcement_enabled"])
            self.assertTrue(graph.last_retrieval_trace["edge_weight_calibration_enabled"])

    def test_frozen_retrieval_is_repeatable_and_does_not_mutate_edges(self):
        with tempfile.TemporaryDirectory() as temp:
            graph = HebbianMemoryGraph(str(Path(temp) / "memory.json"))
            graph.nodes = {
                "0": {"id": "0", "content": "a", "embedding": [1.0, 0.0], "timestamp": "2026-01-01", "keywords": []},
                "1": {"id": "1", "content": "b", "embedding": [0.8, 0.2], "timestamp": "2026-01-01", "keywords": []},
                "2": {"id": "2", "content": "c", "embedding": [0.0, 1.0], "timestamp": "2026-01-01", "keywords": []},
            }
            graph.add_edge("0", "2", weight=0.5, bidirectional=True)
            before = copy.deepcopy(graph.edges)
            kwargs = dict(
                top_k=2,
                query_keywords_override=set(),
                query_embedding_override=np.array([1.0, 0.0]),
                current_time_override="2026-01-02 00:00:00",
                update_graph=False,
            )
            with patch("hela_mem.hebbian_memory.compute_time_decay", lambda *args: 1.0):
                first = graph.retrieve("query", **kwargs)
                first_trace = copy.deepcopy(graph.last_retrieval_trace)
                second = graph.retrieve("query", **kwargs)
                second_trace = copy.deepcopy(graph.last_retrieval_trace)
            self.assertEqual([row["node"]["id"] for row in first], [row["node"]["id"] for row in second])
            self.assertEqual([row["score"] for row in first], [row["score"] for row in second])
            self.assertEqual(first_trace["final_scores"], second_trace["final_scores"])
            self.assertEqual(before, graph.edges)
            self.assertFalse(second_trace["reinforcement_enabled"])

    def test_default_and_explicit_update_graph_preserve_reinforcement(self):
        def make_graph(path):
            graph = HebbianMemoryGraph(str(path))
            graph.nodes = {
                "0": {"id": "0", "content": "a", "embedding": [1.0, 0.0], "timestamp": "2026-01-01", "keywords": []},
                "1": {"id": "1", "content": "b", "embedding": [0.9, 0.1], "timestamp": "2026-01-01", "keywords": []},
            }
            return graph

        with tempfile.TemporaryDirectory() as temp:
            default_graph = make_graph(Path(temp) / "default.json")
            explicit_graph = make_graph(Path(temp) / "explicit.json")
            kwargs = dict(
                top_k=2,
                query_keywords_override=set(),
                query_embedding_override=np.array([1.0, 0.0]),
                current_time_override="2026-01-02 00:00:00",
            )
            with patch("hela_mem.hebbian_memory.compute_time_decay", lambda *args: 1.0):
                default_results = default_graph.retrieve("query", **kwargs)
                explicit_results = explicit_graph.retrieve("query", update_graph=True, **kwargs)
            self.assertEqual(
                [row["node"]["id"] for row in default_results],
                [row["node"]["id"] for row in explicit_results],
            )
            self.assertEqual(default_graph.edges, explicit_graph.edges)
            self.assertGreater(default_graph.edges["0"]["1"], 0.0)
            self.assertTrue(default_graph.last_retrieval_trace["reinforcement_enabled"])

    def test_five_items_resume_and_eval_schema(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            encoded, results = root / "encoded", root / "eval_results"
            encoded.mkdir()
            patches = (
                patch("hela_mem.hebbian_memory.get_embedding", fake_embedding),
                patch("hela_mem.hebbian_memory.llm_extract_keywords", lambda text: {"tea"}),
                patch("hela_mem.hebbian_knowledge_memory.get_embedding", fake_embedding),
            )
            with patches[0], patches[1], patches[2]:
                with ThreadPoolExecutor(max_workers=4) as executor:
                    list(executor.map(lambda value: enc.encode_single_item(value, str(encoded), FakeExtractionClient()), [item(i) for i in range(5)]))
                mtimes = {path.name: path.stat().st_mtime_ns for path in encoded.glob("*_hebbian.json")}
                self.assertTrue(all(enc._complete(str(encoded), f"q{i}") for i in range(5)))
                self.assertEqual(mtimes, {path.name: path.stat().st_mtime_ns for path in encoded.glob("*_hebbian.json")})

                def fake_generate(prompt, messages, model=None, max_retries=3, role="generation"):
                    return "yes" if role == "judge" else "<think>hidden</think>tea"

                with patch.object(ev, "gpt_generate_answer_with_rotation", fake_generate):
                    result = ev.evaluate_single_item(item(0), 0, str(encoded), results_dir=str(results))
                self.assertEqual(result["status"], "ok")
                self.assertEqual(result["prediction"], "<think>hidden</think>tea")
                self.assertIn("retrieved_episodic", result)
                self.assertIn("judge_model", result)
                saved = json.loads((results / "result_q0.json").read_text(encoding="utf-8"))
                self.assertEqual(saved["question_id"], "q0")


if __name__ == "__main__":
    unittest.main()
