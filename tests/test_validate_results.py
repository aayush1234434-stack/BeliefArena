"""Tests for deduplication, validation, purge, and frozen dataset contract."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from results_schema import CONDITION_NAMES, is_valid_row
from validate_results import (
    DEFAULT_CLEAN_PATH,
    dedupe_rows,
    keys_to_purge,
    load_question_ids,
    purge_invalid_rows,
    sort_rows,
    validate_manifest_contract,
    validate_rows,
)

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = ROOT / "data" / "final" / "manifest.json"
ARCHIVED_CLEAN_PATH = ROOT / "data" / "final" / "results_clean_v1.0.0_invalid.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def valid_row(**overrides) -> dict:
    row = {
        "model": "m",
        "question_id": "Q01",
        "condition": "prior",
        "answer": "No",
        "confidence": "8",
        "prior_answer": "No",
        "prior_confidence": "8",
    }
    row.update(overrides)
    return row


class TestDeduplication(unittest.TestCase):
    def test_dedupe_keep_last(self):
        rows = [
            {"model": "m", "question_id": "Q01", "condition": "prior", "answer": "No", "v": 1},
            {"model": "m", "question_id": "Q01", "condition": "prior", "answer": "Yes", "v": 2},
            {"model": "m", "question_id": "Q01", "condition": "strength_alone", "answer": "No", "v": 1},
        ]
        deduped, dup_count = dedupe_rows(rows, keep="last")
        self.assertEqual(dup_count, 1)
        self.assertEqual(len(deduped), 2)
        prior = next(r for r in deduped if r["condition"] == "prior")
        self.assertEqual(prior["v"], 2)

    def test_dedupe_keep_first(self):
        rows = [
            {"model": "m", "question_id": "Q01", "condition": "prior", "v": 1},
            {"model": "m", "question_id": "Q01", "condition": "prior", "v": 2},
        ]
        deduped, dup_count = dedupe_rows(rows, keep="first")
        self.assertEqual(dup_count, 1)
        self.assertEqual(deduped[0]["v"], 1)

    def test_sort_rows_stable_order(self):
        rows = [
            {"model": "b", "question_id": "Q02", "condition": "prior"},
            {"model": "a", "question_id": "Q01", "condition": "strength_alone"},
            {"model": "a", "question_id": "Q01", "condition": "prior"},
        ]
        sorted_rows = sort_rows(rows)
        self.assertEqual(sorted_rows[0]["model"], "a")
        self.assertEqual(sorted_rows[0]["condition"], "prior")
        self.assertEqual(sorted_rows[1]["condition"], "strength_alone")


class TestValidation(unittest.TestCase):
    def test_valid_minimal_row(self):
        rows = [
            {
                "model": "test/model",
                "question_id": "Q01",
                "condition": condition,
                "answer": "No",
                "confidence": "8",
                "prior_answer": "No",
                "prior_confidence": "8",
            }
            for condition in CONDITION_NAMES
        ]
        errors = validate_rows(rows, {"Q01"}, expected_questions=1)
        self.assertEqual(errors, [])

    def test_rejects_duplicate_key(self):
        row = valid_row()
        errors = validate_rows([row, row], {"Q01"}, expected_questions=1)
        self.assertTrue(any("duplicate" in error for error in errors))

    def test_rejects_invalid_confidence(self):
        row = valid_row(confidence="0", prior_confidence="0")
        errors = validate_rows([row], {"Q01"}, expected_questions=1)
        self.assertTrue(any("confidence" in error for error in errors))


class TestPurgeInvalid(unittest.TestCase):
    def test_purges_invalid_prior_and_dependent_conditions(self):
        prior = valid_row(model="llama", condition="prior", confidence="0")
        strength = valid_row(
            model="llama",
            condition="strength_alone",
            confidence="8",
            prior_confidence="0",
        )
        purge = keys_to_purge([prior, strength])
        self.assertIn(("llama", "Q01", "prior"), purge)
        self.assertIn(("llama", "Q01", "strength_alone"), purge)

    def test_purges_only_invalid_rows_when_prior_ok(self):
        prior = valid_row(model="llama", condition="prior")
        bad = valid_row(model="llama", condition="majority_vs_credibility", confidence="0")
        cleaned, removed = purge_invalid_rows([prior, bad])
        self.assertEqual(removed, 1)
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(cleaned[0]["condition"], "prior")


class TestArchivedFrozenDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        cls.archived = cls.manifest["archived_releases"][0]
        cls.rows = load_jsonl(ARCHIVED_CLEAN_PATH)

    def test_archived_file_exists(self):
        self.assertTrue(ARCHIVED_CLEAN_PATH.exists())

    def test_archived_shape(self):
        self.assertEqual(len(self.rows), self.archived["row_count"])

    def test_archived_fails_strict_validation_as_documented(self):
        question_ids = load_question_ids()
        errors = validate_rows(self.rows, question_ids)
        self.assertEqual(len(errors), self.archived["error_count_at_freeze"])
        self.assertGreater(self.archived["invalid_confidence_zero_rows"], 0)


class TestFrozenDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    def setUp(self):
        if not DEFAULT_CLEAN_PATH.exists():
            self.skipTest(
                "results_clean.jsonl not frozen yet; run scripts/repair_and_freeze.sh"
            )
        self.rows = load_jsonl(DEFAULT_CLEAN_PATH)

    def test_manifest_requires_strict_validation_when_frozen(self):
        self.assertTrue(self.manifest["validation"]["passes_strict_validation"])

    def test_manifest_matches_file(self):
        self.assertEqual(len(self.rows), self.manifest["row_count"])
        self.assertEqual(
            sorted({row["model"] for row in self.rows}),
            sorted(self.manifest["models"]),
        )

    def test_no_duplicate_keys_in_frozen_file(self):
        keys = [(r["model"], r["question_id"], r["condition"]) for r in self.rows]
        self.assertEqual(len(keys), len(set(keys)))

    def test_expected_shape(self):
        for model in self.manifest["models"]:
            for condition in CONDITION_NAMES:
                count = sum(
                    1
                    for row in self.rows
                    if row["model"] == model and row["condition"] == condition
                )
                self.assertEqual(count, self.manifest["questions_per_model"])

    def test_passes_strict_validation(self):
        question_ids = load_question_ids()
        errors = validate_rows(self.rows, question_ids)
        self.assertEqual(errors, [])
        self.assertTrue(all(is_valid_row(row) for row in self.rows))


class TestFrozenManifestContract(unittest.TestCase):
    def test_frozen_state_documented(self):
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"], "frozen")
        self.assertTrue(manifest["validation"]["passes_strict_validation"])
        self.assertTrue(manifest["validation"]["primary_file_present"])
        self.assertEqual(manifest["row_count"], 1600)

    def test_contract_validation_passes(self):
        errors = validate_manifest_contract(MANIFEST_PATH)
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
