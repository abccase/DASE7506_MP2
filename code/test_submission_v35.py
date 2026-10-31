"""Behavioural tests for the submitted artefact.

Covers the two components this work adds to the supplied pipeline -- the
arithmetic verifier and the answer-validity rule -- plus the inference contract
(one integer index per question, only parsable candidates eligible).
"""
import unittest
from pathlib import Path

import numpy as np

from common import load_inputs, parse_answer
from submission_v35 import SCHEMA, predict
from verifier_v31 import COLUMNS, build_rows, extract, normalise, safe_eval


class TestVerifier(unittest.TestCase):
    def test_latex_operators(self):
        self.assertEqual(normalise("2 \\times 3 = 6"), "2 * 3 = 6")
        self.assertIn("((50)/(5))", normalise("\\frac{50}{5} = 10"))

    def test_safe_eval_rejects_variables(self):
        self.assertIsNone(safe_eval("2*x"))
        self.assertAlmostEqual(safe_eval("2+3*4"), 14.0)

    def test_true_chain(self):
        f = extract("Step 1:\n50 - 20 = 30\nStep 2:\n30 / 3 = 10\n", "10")
        self.assertEqual((f[0], f[1], f[3], f[4]), (2, 0, 0, 1))

    def test_false_equation_flagged(self):
        f = extract("30 + 20 = 60\n", "60")
        self.assertEqual((f[0], f[1], f[3]), (1, 1, 1))

    def test_symbolic_step_not_a_false_positive(self):
        f = extract("x + 60 = 15 + 60 = 75\n", "75")
        self.assertEqual(f[1], 0)

    def test_shape(self):
        bank = [{"candidates": [{"text": "1 + 1 = 2\n"}] * 4}]
        self.assertEqual(build_rows(bank).shape, (1, 4, len(COLUMNS)))


class TestSubmittedInference(unittest.TestCase):
    bundle = Path(__file__).resolve().parent.parent / "release" / "verifier-v35" / "model_bundle"

    def test_requires_checkpoint(self):
        bank, arrays = load_inputs("validation")
        with self.assertRaises(ValueError):
            predict(bank, arrays, None)

    def test_one_index_per_question_and_rules_respected(self):
        if not (self.bundle / "model.npz").exists():
            self.skipTest("submitted bundle not present")
        bank, arrays = load_inputs("validation")
        with np.load(self.bundle / "model.npz", allow_pickle=False) as d:
            self.assertEqual(str(d["feature_schema"].item()), SCHEMA)
        choices = predict(bank, arrays, self.bundle)
        self.assertEqual(len(choices), len(bank))
        self.assertTrue(all(type(c) is int and 0 <= c < 4 for c in choices))
        from fractions import Fraction
        for q, c in enumerate(choices):
            parsed = [parse_answer(x["text"]) for x in bank[q]["candidates"]]
            self.assertIsNotNone(parsed[c], "selected an unparsable candidate")
            legal = [p for p in parsed if p is not None and Fraction(p).denominator == 1 and Fraction(p) != 0]
            if legal:
                self.assertIn(parsed[c], legal, "selected an answer the rule forbids")


if __name__ == "__main__":
    unittest.main()
