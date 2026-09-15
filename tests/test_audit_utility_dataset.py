import unittest

from hela_mem.audit_utility_dataset import describe, transition_sanity


class UtilityAuditTest(unittest.TestCase):
    def test_transition_sanity_reports_expected_direction(self):
        rows = [
            {"transition": "W2C", "utility_score": .4},
            {"transition": "W2C", "utility_score": .2},
            {"transition": "C2W", "utility_score": -.3},
        ]
        result = transition_sanity(rows)
        self.assertEqual(result["W2C"]["expected_sign_rate"], 1.0)
        self.assertEqual(result["C2W"]["expected_sign_rate"], 1.0)

    def test_describe_is_numeric_and_empty_safe(self):
        self.assertEqual(describe([]), {"count": 0})
        self.assertEqual(describe([-.2, .2])["median"], 0.0)


if __name__ == "__main__":
    unittest.main()
