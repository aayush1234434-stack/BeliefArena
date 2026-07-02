#!/usr/bin/env python3
"""Validate and normalize question.json formatting."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

QUESTIONS_FILE = Path("question.json")

REQUIRED_FIELDS = {
    "question_id",
    "correct_answer",
    "topic",
    "strength_correct",
    "strength_wrong",
    "majority_vague_correct",
    "majority_vague_wrong",
    "credibility_correct",
    "credibility_wrong",
    "majority_plain_correct",
    "majority_plain_wrong",
}


def extract_question_objects(raw: str) -> list[dict]:
    """Salvage question dicts from malformed JSON by brace matching."""
    questions: list[dict] = []
    pattern = re.compile(r'\{\s*"question_id"\s*:\s*"([^"]+)"')

    for match in pattern.finditer(raw):
        start = match.start()
        depth = 0
        in_string = False
        escape = False
        end = None

        for i in range(start, len(raw)):
            ch = raw[i]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break

        if end is None:
            continue
        try:
            obj = json.loads(raw[start:end])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "question_id" in obj:
            questions.append(obj)

    return questions


def dedupe_questions(questions: list[dict]) -> list[dict]:
    """Keep the first occurrence of each question_id."""
    seen: set[str] = set()
    unique: list[dict] = []
    for question in questions:
        qid = question["question_id"]
        if qid in seen:
            continue
        seen.add(qid)
        unique.append(question)
    return unique


def sort_questions(questions: list[dict]) -> list[dict]:
    def key(question: dict) -> tuple[int, str]:
        qid = question["question_id"]
        num = int(qid[1:]) if qid.startswith("Q") and qid[1:].isdigit() else 10**9
        return num, qid

    return sorted(questions, key=key)


def validate_questions(questions: list[dict]) -> list[str]:
    issues: list[str] = []
    for question in questions:
        qid = question.get("question_id", "?")
        missing = REQUIRED_FIELDS - set(question.keys())
        if missing:
            issues.append(f"{qid}: missing fields {sorted(missing)}")
        for field in ("majority_vague_correct", "majority_vague_wrong", "majority_plain_correct", "majority_plain_wrong"):
            value = question.get(field)
            if not isinstance(value, list) or len(value) != 10:
                count = len(value) if isinstance(value, list) else "not a list"
                issues.append(f"{qid}: {field} should have 10 items, has {count}")
        if "question_text" in question:
            issues.append(f"{qid}: question_text should live in question_texts.json, not question.json")
    return issues


def load_questions(path: Path = QUESTIONS_FILE) -> list[dict]:
    raw = path.read_text(encoding="utf-8")
    try:
        data = json.loads(raw)
        if not isinstance(data, list):
            raise ValueError("question.json must be a JSON array.")
        return data
    except json.JSONDecodeError as exc:
        print(f"Invalid JSON ({exc}); attempting salvage extraction...")
        salvaged = extract_question_objects(raw)
        if not salvaged:
            raise
        print(f"Salvaged {len(salvaged)} question object(s) from malformed file.")
        return salvaged


def normalize_questions(path: Path = QUESTIONS_FILE) -> list[dict]:
    questions = dedupe_questions(sort_questions(load_questions(path)))
    issues = validate_questions(questions)
    if issues:
        print("Validation issues:")
        for issue in issues:
            print(f"  - {issue}")
        raise SystemExit(1)

    canonical = json.dumps(questions, indent=2, ensure_ascii=False) + "\n"
    path.write_text(canonical, encoding="utf-8")
    return questions


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else QUESTIONS_FILE
    questions = normalize_questions(target)
    print(f"Normalized {target} with {len(questions)} question(s): {[q['question_id'] for q in questions]}")
