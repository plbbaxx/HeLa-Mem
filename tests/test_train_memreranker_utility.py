import unittest
from collections import Counter

from hela_mem.train_memreranker_utility import (
    OFFICIAL_PREFIX,
    OFFICIAL_SUFFIX,
    PromptEncoder,
    QuestionBalancedSampler,
    format_query,
    ndcg,
    resolve_yes_no_token_ids,
    spearman,
)


class FakeTokenizer:
    unk_token_id = -1
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        if text == "yes":
            return [7]
        if text == "no":
            return [8]
        return list(text.encode())

    def convert_tokens_to_ids(self, token):
        return {"yes": 7, "no": 8}.get(token, -1)


class MemRerankerUtilityTrainingTest(unittest.TestCase):
    def test_official_yes_no_tokens_are_resolved_not_guessed(self):
        self.assertEqual(resolve_yes_no_token_ids(FakeTokenizer()), (7, 8))

    def test_prompt_uses_frozen_base_only_when_conditioned(self):
        row = {"question": "Where?", "candidate_text": "At school", "actual_baseline_context": "BASE15"}
        self.assertNotIn("BASE15", format_query(row, False))
        self.assertIn("BASE15", format_query(row, True))

    def test_encoder_keeps_official_prefix_suffix_and_reports_truncation(self):
        tokenizer = FakeTokenizer()
        minimum = len(tokenizer.encode(OFFICIAL_PREFIX)) + len(tokenizer.encode(OFFICIAL_SUFFIX))
        encoder = PromptEncoder(tokenizer, minimum + 20)
        encoded = encoder.encode({"question": "q" * 100, "candidate_text": "c", "actual_baseline_context": "b"}, True)
        self.assertTrue(encoded.truncated)
        self.assertEqual(encoded.input_ids[:len(encoder.prefix)], encoder.prefix)
        self.assertEqual(encoded.input_ids[-len(encoder.suffix):], encoder.suffix)

    def test_question_balanced_sampler_covers_only_pair_questions_uniformly(self):
        pairs = ([{"question_id": "a"}] * 8) + ([{"question_id": "b"}] * 2)
        sampled = list(QuestionBalancedSampler(pairs, seed=3))
        counts = Counter(pairs[index]["question_id"] for index in sampled)
        self.assertEqual(counts, {"a": 5, "b": 5})

    def test_ranking_metrics(self):
        self.assertAlmostEqual(spearman([1, 2, 3], [10, 20, 30]), 1.0)
        self.assertAlmostEqual(ndcg([1, 2, 3], [10, 20, 30]), 1.0)


if __name__ == "__main__":
    unittest.main()
