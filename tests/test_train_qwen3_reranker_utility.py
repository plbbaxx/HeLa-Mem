import importlib.util
import unittest
from collections import Counter
from types import SimpleNamespace

from hela_mem.train_qwen3_reranker_utility import (
    BaseTailTruncatingEncoder,
    FullQuestionBalancedSampler,
    choose_max_length,
    forward_shape_diagnostic,
    reciprocal_rank,
    score_batch,
    trainable_parameter_report,
)


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(text.encode("utf-8"))


class FakeParameter:
    def __init__(self, count, requires_grad):
        self._count = count
        self.requires_grad = requires_grad
        self.shape = (count,)

    def numel(self):
        return self._count


class FakeModel:
    def __init__(self, rows):
        self.rows = rows

    def named_parameters(self):
        return iter(self.rows)


class TinyDecoder:
    def __call__(self, input_ids, attention_mask, **kwargs):
        import torch
        hidden = torch.nn.functional.one_hot(input_ids % 3, num_classes=3).float()
        return SimpleNamespace(last_hidden_state=hidden)


class TinyCausalLM:
    def __init__(self):
        import torch
        self.model = TinyDecoder()
        self.lm_head = torch.nn.Linear(3, 5, bias=False)
        with torch.no_grad():
            self.lm_head.weight.copy_(torch.arange(15, dtype=torch.float32).reshape(5, 3))

    def __call__(self, *args, **kwargs):
        raise AssertionError("full CausalLM forward must not be called")

    def get_output_embeddings(self):
        return self.lm_head

    def parameters(self):
        return self.lm_head.parameters()


class QwenUtilityTrainingTest(unittest.TestCase):
    def test_adaptive_length_policy(self):
        self.assertEqual(choose_max_length([3000] * 90 + [5000] * 10)[0], 4096)
        self.assertEqual(choose_max_length([5000] * 95 + [9000] * 5)[0], 8192)
        self.assertEqual(choose_max_length([5000] * 80 + [10000] * 20)[0], 16384)

    def test_encoder_preserves_question_candidate_and_drops_low_rank_base(self):
        tokenizer = FakeTokenizer()
        row = {
            "question": "QUESTION_UNIQUE",
            "candidate_text": "CANDIDATE_UNIQUE",
            "actual_baseline_context": (
                "[Direct Match | Relevancy: 1.00]\nTime: t\nContent: HIGH_RANK\n\n"
                "[Associative Memory | Relevancy: 0.10]\nTime: t\nContent: LOW_RANK" 
            ),
        }
        raw = BaseTailTruncatingEncoder(tokenizer, 100000).encode(row, True)
        limited = BaseTailTruncatingEncoder(tokenizer, raw.raw_token_count - 20).encode(row, True)
        text = bytes(limited.input_ids).decode("utf-8")
        self.assertIn("QUESTION_UNIQUE", text)
        self.assertIn("CANDIDATE_UNIQUE", text)
        self.assertIn("HIGH_RANK", text)
        self.assertNotIn("LOW_RANK", text)
        self.assertEqual(limited.base_blocks_kept, 1)

    def test_full_sampler_is_question_balanced_but_keeps_full_epoch_size(self):
        pairs = ([{"question_id": "a"}] * 8) + ([{"question_id": "b"}] * 2)
        sampled = list(FullQuestionBalancedSampler(pairs, seed=1, epoch=1))
        counts = Counter(pairs[index]["question_id"] for index in sampled)
        self.assertEqual(len(sampled), len(pairs))
        self.assertEqual(counts, {"a": 5, "b": 5})

    def test_reciprocal_rank_targets_max_utility_candidate(self):
        self.assertEqual(reciprocal_rank([.9, .8, .7], [0, 2, 1]), .5)

    def test_parameter_audit_accepts_only_lora_trainables(self):
        report = trainable_parameter_report(FakeModel([
            ("base.weight", FakeParameter(1000, False)),
            ("q_proj.lora_A.default.weight", FakeParameter(16, True)),
            ("q_proj.lora_B.default.weight", FakeParameter(16, True)),
        ]))
        self.assertTrue(report["lora_only"])
        self.assertEqual(report["trainable_parameters"], 32)

    def test_parameter_audit_rejects_trainable_backbone(self):
        with self.assertRaisesRegex(RuntimeError, "PEFT freeze invariant failed"):
            trainable_parameter_report(FakeModel([
                ("base.weight", FakeParameter(1000, True)),
                ("q_proj.lora_A.default.weight", FakeParameter(16, True)),
            ]))

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is not installed in the lightweight test environment")
    def test_score_batch_projects_only_yes_no_rows_from_final_hidden(self):
        import torch
        model = TinyCausalLM()
        batch = {
            "input_ids": torch.tensor([[0, 1], [1, 2]]),
            "attention_mask": torch.ones((2, 2), dtype=torch.long),
        }
        scores = score_batch(model, batch, yes_id=3, no_id=1)
        expected = torch.tensor([6.0, 6.0])
        torch.testing.assert_close(scores, expected)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is not installed in the lightweight test environment")
    def test_forward_diagnostic_reports_avoided_vocab_tensor(self):
        import torch
        model = TinyCausalLM()
        batch = {
            "input_ids": torch.tensor([[0, 1, 2], [0, 0, 2]]),
            "attention_mask": torch.tensor([[0, 1, 1], [1, 1, 1]]),
        }
        report = forward_shape_diagnostic(model, batch)
        self.assertEqual(report["sequence_token_lengths"], [2, 3])
        self.assertEqual(report["avoided_full_logits_shape"], [2, 3, 5])
        self.assertEqual(report["actual_projected_logits_shape"], [2, 2])


if __name__ == "__main__":
    unittest.main()
