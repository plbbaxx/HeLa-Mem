import unittest
from collections import Counter

from hela_mem.train_qwen3_reranker_utility import (
    BaseTailTruncatingEncoder,
    FullQuestionBalancedSampler,
    choose_max_length,
    reciprocal_rank,
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


if __name__ == "__main__":
    unittest.main()
