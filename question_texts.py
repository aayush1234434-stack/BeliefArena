"""Extract question texts from question.json into question_texts.json."""

from __future__ import annotations

import json
from pathlib import Path

QUESTIONS_FILE = Path("question.json")
QUESTION_TEXTS_FILE = Path("question_texts.json")


def load_question_texts() -> dict[str, str]:
    if not QUESTION_TEXTS_FILE.exists():
        return {}
    with QUESTION_TEXTS_FILE.open(encoding="utf-8") as handle:
        return json.load(handle)


def sync_question_texts(remove_from_questions: bool = True) -> list[str]:
    """Extract new question_text entries into question_texts.json."""
    with QUESTIONS_FILE.open(encoding="utf-8") as handle:
        questions = json.load(handle)

    question_texts = load_question_texts()
    added: list[str] = []
    stripped = False

    for question in questions:
        qid = question.get("question_id")
        text = question.get("question_text")
        if not qid or not text:
            continue
        if qid not in question_texts:
            question_texts[qid] = text.strip()
            added.append(qid)
        if remove_from_questions and "question_text" in question:
            question.pop("question_text", None)
            stripped = True

    if added:
        with QUESTION_TEXTS_FILE.open("w", encoding="utf-8") as handle:
            json.dump(question_texts, handle, indent=2, ensure_ascii=False)
            handle.write("\n")

    if remove_from_questions and stripped:
        with QUESTIONS_FILE.open("w", encoding="utf-8") as handle:
            json.dump(questions, handle, indent=2, ensure_ascii=False)
            handle.write("\n")

    return added


def enrich_questions(questions: list[dict]) -> list[dict]:
    """Sync any new texts, then attach saved question_text to each question."""
    sync_question_texts()
    texts = load_question_texts()

    enriched = []
    missing = []
    for question in questions:
        qid = question["question_id"]
        text = texts.get(qid)
        if text is None:
            missing.append(qid)
            continue
        row = dict(question)
        row["question_text"] = text
        enriched.append(row)

    if missing:
        raise KeyError(
            "Missing question_text for: "
            + ", ".join(missing)
            + f". Add question_text to {QUESTIONS_FILE} and run again."
        )
    return enriched


def missing_text_ids(questions: list[dict], texts: dict[str, str]) -> list[str]:
    """Return question_ids present in questions but not in the texts registry."""
    return [q["question_id"] for q in questions if q["question_id"] not in texts]


def print_sync_status(questions: list[dict], texts: dict[str, str], added: list[str]) -> None:
    """Print a human-readable sync summary."""
    pending = [
        q["question_id"]
        for q in questions
        if q.get("question_text") and q["question_id"] not in texts
    ]
    missing = missing_text_ids(questions, texts)

    print(f"Questions in {QUESTIONS_FILE}: {len(questions)}")
    print(f"Texts in {QUESTION_TEXTS_FILE}: {len(texts)}")
    print(f"Added this run: {added if added else 'none'}")

    if pending:
        print(f"Ready to extract (have question_text in {QUESTIONS_FILE}): {pending}")

    if missing:
        print(
            "Missing question_text (add a question_text field in question.json, save, then re-run): "
            + ", ".join(missing)
        )
    elif len(questions) == len(texts):
        print("All questions have saved texts.")


if __name__ == "__main__":
    with QUESTIONS_FILE.open(encoding="utf-8") as handle:
        questions = json.load(handle)
    added = sync_question_texts()
    texts = load_question_texts()
    print_sync_status(questions, texts, added)
