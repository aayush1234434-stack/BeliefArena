"""Tests for answer/confidence parsing and outcome computation."""

from __future__ import annotations

import unittest

from results_schema import (
    compute_flipped,
    compute_trial_outcomes,
    compute_winner,
    extract_confidence,
    is_valid_answer,
    is_valid_confidence,
    normalize_answer,
    parse_confidence,
    parse_model_response,
    parse_validated_model_response,
)


class TestAnswerParsing(unittest.TestCase):
    def test_normalize_yes_no_variants(self):
        self.assertEqual(normalize_answer("Yes"), "yes")
        self.assertEqual(normalize_answer("Yes,"), "yes")
        self.assertEqual(normalize_answer("NO."), "no")
        self.assertEqual(normalize_answer("no!"), "no")

    def test_invalid_answers(self):
        self.assertEqual(normalize_answer("Maybe"), "maybe")
        self.assertIsNone(normalize_answer(None))
        self.assertFalse(is_valid_answer("Maybe"))
        self.assertTrue(is_valid_answer("Yes"))


class TestConfidenceParsing(unittest.TestCase):
    def test_valid_confidence(self):
        self.assertEqual(parse_confidence("7"), 7.0)
        self.assertEqual(parse_confidence("2."), 2.0)
        self.assertTrue(is_valid_confidence("10"))

    def test_invalid_confidence_not_clamped(self):
        self.assertIsNone(parse_confidence("11"))
        self.assertIsNone(parse_confidence("0"))
        self.assertEqual(extract_confidence("11"), 11.0)
        self.assertEqual(extract_confidence("0"), 0.0)
        self.assertFalse(is_valid_confidence("11"))
        self.assertFalse(is_valid_confidence("0"))

    def test_parse_model_response(self):
        self.assertEqual(parse_model_response("No 7"), ("No", "7"))
        self.assertEqual(parse_model_response("Yes, 9"), ("Yes,", "9"))

    def test_parse_validated_model_response(self):
        self.assertEqual(parse_validated_model_response("No 7"), ("No", "7"))
        self.assertEqual(parse_validated_model_response("Yes, 9"), ("Yes", "9"))
        self.assertIsNone(parse_validated_model_response("No 0"))
        self.assertIsNone(parse_validated_model_response("Maybe 5"))
        self.assertIsNone(parse_validated_model_response("No"))
        self.assertIsNone(parse_validated_model_response("No 11"))


class TestOutcomeComputation(unittest.TestCase):
    def test_flip_ignores_punctuation(self):
        self.assertTrue(compute_flipped("Yes,", "No"))
        self.assertFalse(compute_flipped("Yes", "Yes,"))
        self.assertFalse(compute_flipped("Maybe", "Yes"))

    def test_winner_competitive_conditions(self):
        self.assertEqual(
            compute_winner("Yes", "Yes", "strength_vs_majority"),
            "strength",
        )
        self.assertEqual(
            compute_winner("No", "Yes", "strength_vs_majority"),
            "majority",
        )
        self.assertIsNone(compute_winner("Yes", "Yes", "prior"))

    def test_winner_with_punctuation(self):
        self.assertEqual(
            compute_winner("Yes,", "No", "majority_vs_credibility"),
            "credibility",
        )

    def test_compute_trial_outcomes(self):
        flipped, winner = compute_trial_outcomes(
            "Yes,",
            "No",
            "Yes",
            "strength_vs_credibility",
        )
        self.assertTrue(flipped)
        self.assertEqual(winner, "strength")


if __name__ == "__main__":
    unittest.main()
