import json
import sys
import tempfile
import types
from concurrent.futures import ThreadPoolExecutor
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

# The workstation has the legacy OpenAI package; production requires openai>=1.
if not hasattr(__import__("openai"), "OpenAI"):
    module = types.ModuleType("openai")
    module.OpenAI = object
    sys.modules["openai"] = module

from hela_mem import encode_longmemeval as enc
from hela_mem import eval_longmemeval as ev
from hela_mem.hebbian_memory import apply_redundancy_aware_inhibition


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
    def test_redundancy_aware_inhibition_changes_competitive_ranking(self):
        scores = np.array([0.82, 0.78, 0.72, 0.68])
        embeddings = np.array([
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ])
        inhibited, penalties = apply_redundancy_aware_inhibition(
            scores, embeddings, gamma=0.3
        )
        self.assertEqual(np.argsort(inhibited)[::-1].tolist(), [0, 2, 1, 3])
        self.assertGreater(penalties[1], penalties[2])

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
