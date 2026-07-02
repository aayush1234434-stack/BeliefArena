"""Tests for collection-time response validation in main.ask_model."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import main  # noqa: E402


def mock_completion(content: str | None) -> MagicMock:
    response = MagicMock()
    response.choices[0].message.content = content
    return response


class TestAskModel(unittest.TestCase):
    def setUp(self):
        main._client = None

    @patch("main.time.sleep")
    @patch("main.throttle")
    @patch("main.get_client")
    def test_retries_confidence_zero(self, mock_get_client, _mock_throttle, _mock_sleep):
        mock_get_client.return_value.chat.completions.create.side_effect = [
            mock_completion("No 0"),
            mock_completion("No 7"),
        ]
        result = main.ask_model({"question_text": "Test?"}, [], label="test")
        self.assertEqual(result["answer"], "No")
        self.assertEqual(result["confidence"], "7")
        self.assertEqual(mock_get_client.return_value.chat.completions.create.call_count, 2)

    @patch("main.time.sleep")
    @patch("main.throttle")
    @patch("main.get_client")
    def test_retries_malformed_answer(self, mock_get_client, _mock_throttle, _mock_sleep):
        mock_get_client.return_value.chat.completions.create.side_effect = [
            mock_completion("Maybe 5"),
            mock_completion("Yes 5"),
        ]
        result = main.ask_model({"question_text": "Test?"}, [], label="test")
        self.assertEqual(result["answer"], "Yes")
        self.assertEqual(result["confidence"], "5")

    @patch("main.time.sleep")
    @patch("main.throttle")
    @patch("main.get_client")
    def test_raises_after_exhausted_retries(self, mock_get_client, _mock_throttle, _mock_sleep):
        mock_get_client.return_value.chat.completions.create.return_value = mock_completion("No 0")
        with self.assertRaisesRegex(RuntimeError, "Failed to get a valid response"):
            main.ask_model({"question_text": "Test?"}, [], label="test", max_retries=2)
        self.assertEqual(mock_get_client.return_value.chat.completions.create.call_count, 2)


if __name__ == "__main__":
    unittest.main()
