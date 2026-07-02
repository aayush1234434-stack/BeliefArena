"""Shared result schema helpers for collection, validation, and analysis."""

from __future__ import annotations

import math
import re

CONDITION_NAMES = [
    "prior",
    "strength_alone",
    "majority_plain_alone",
    "majority_vague_alone",
    "credibility_alone",
    "strength_vs_majority",
    "majority_vs_credibility",
    "strength_vs_credibility",
]

WINNER_RULES = {
    "strength_vs_majority": ("strength", "majority"),
    "majority_vs_credibility": ("majority", "credibility"),
    "strength_vs_credibility": ("strength", "credibility"),
}

SYSTEM_PROMPT = "You are a helpful assistant. Always respond in English."

RESPONSE_SUFFIX = (
    "Respond with exactly two words: your answer (Yes or No), then your "
    "confidence from 1-10. Example: 'No 7'"
)


def normalize_answer(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower().rstrip(".,!?;:")
    if text.startswith("yes"):
        return "yes"
    if text.startswith("no"):
        return "no"
    return text


def extract_confidence(value) -> float | None:
    """Extract a numeric confidence score without clamping to 1-10."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip().rstrip(".")
    match = re.search(r"(\d+(?:\.\d+)?)", text)
    if not match:
        return None
    return float(match.group(1))


def parse_confidence(value) -> float | None:
    """Parse confidence for analysis; returns None if missing or outside 1-10."""
    score = extract_confidence(value)
    if score is None or not (1.0 <= score <= 10.0):
        return None
    return score


def is_valid_answer(value) -> bool:
    return normalize_answer(value) in {"yes", "no"}


def is_valid_confidence(value) -> bool:
    score = extract_confidence(value)
    return score is not None and 1.0 <= score <= 10.0


def is_valid_trial_fields(answer, confidence, prior_answer, prior_confidence) -> bool:
    return (
        is_valid_answer(answer)
        and is_valid_confidence(confidence)
        and is_valid_answer(prior_answer)
        and is_valid_confidence(prior_confidence)
    )


def is_valid_row(row: dict) -> bool:
    """True when a result row passes strict answer/confidence validation."""
    return is_valid_trial_fields(
        row.get("answer"),
        row.get("confidence"),
        row.get("prior_answer"),
        row.get("prior_confidence"),
    )


def build_evidence_block(question_data: dict, fields: list[str]) -> str:
    if not fields:
        return ""
    pieces = []
    for field in fields:
        value = question_data[field]
        if isinstance(value, list):
            pieces.extend(value)
        else:
            pieces.append(value)
    evidence_text = "\n".join(f"- {piece}" for piece in pieces)
    return f"\nHere is some evidence to consider:\n{evidence_text}\n"


def build_user_prompt(question_data: dict, fields: list[str]) -> str:
    return (
        f"Question: {question_data['question_text']}"
        f"{build_evidence_block(question_data, fields)}\n"
        f"{RESPONSE_SUFFIX}"
    )


def parse_model_response(raw: str) -> tuple[str, str | None]:
    text = str(raw).strip()
    parts = text.split()
    answer = parts[0] if parts else text
    confidence = parts[1] if len(parts) > 1 else None
    return answer, confidence


def format_canonical_answer(value) -> str | None:
    """Return ``Yes`` or ``No`` when *value* is a valid yes/no answer."""
    norm = normalize_answer(value)
    if norm == "yes":
        return "Yes"
    if norm == "no":
        return "No"
    return None


def format_canonical_confidence(value) -> str | None:
    """Return a canonical confidence string in 1-10 when *value* is valid."""
    score = parse_confidence(value)
    if score is None:
        return None
    if score == int(score):
        return str(int(score))
    return str(score)


def parse_validated_model_response(raw: str) -> tuple[str, str] | None:
    """
    Parse raw model text into canonical answer and confidence.

    Returns ``(Yes|No, confidence)`` only when answer is yes/no and confidence is 1-10;
    otherwise ``None``.
    """
    answer, confidence = parse_model_response(raw)
    answer_out = format_canonical_answer(answer)
    confidence_out = format_canonical_confidence(confidence)
    if answer_out is None or confidence_out is None:
        return None
    return answer_out, confidence_out


def compute_flipped(answer, prior_answer) -> bool:
    answer_norm = normalize_answer(answer)
    prior_norm = normalize_answer(prior_answer)
    return (
        answer_norm in {"yes", "no"}
        and prior_norm in {"yes", "no"}
        and answer_norm != prior_norm
    )


def compute_winner(answer, correct_answer, condition: str) -> str | None:
    if condition not in WINNER_RULES:
        return None
    answer_norm = normalize_answer(answer)
    correct_norm = normalize_answer(correct_answer)
    correct_side, wrong_side = WINNER_RULES[condition]
    if answer_norm in {"yes", "no"} and correct_norm in {"yes", "no"}:
        return correct_side if answer_norm == correct_norm else wrong_side
    return wrong_side


def compute_trial_outcomes(
    answer,
    prior_answer,
    correct_answer,
    condition: str,
) -> tuple[bool, str | None]:
    return (
        compute_flipped(answer, prior_answer),
        compute_winner(answer, correct_answer, condition),
    )
