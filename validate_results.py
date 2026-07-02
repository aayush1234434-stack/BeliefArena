#!/usr/bin/env python3
"""Validate experiment result files and optionally dedupe or freeze a clean snapshot."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from results_io import (
    LEGACY_RESULTS_PATH,
    RESULTS_DIR,
    discover_result_files,
    load_jsonl_rows,
    model_results_path,
)
from results_schema import CONDITION_NAMES, is_valid_answer, is_valid_confidence, is_valid_row

ROOT = Path(__file__).resolve().parent
QUESTION_JSON = ROOT / "question.json"
DEFAULT_CLEAN_PATH = ROOT / "data" / "final" / "results_clean.jsonl"
DEFAULT_MANIFEST_PATH = ROOT / "data" / "final" / "manifest.json"
EXPECTED_QUESTIONS = 100


def load_question_ids(path: Path = QUESTION_JSON) -> set[str]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    return {row["question_id"] for row in data}


def load_jsonl_file(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_manifest(path: Path = DEFAULT_MANIFEST_PATH) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def validate_archived_release(
    rows: list[dict],
    release: dict,
    manifest: dict,
    question_ids: set[str],
) -> list[str]:
    """Validate an archived frozen file against its documented contract."""
    errors: list[str] = []
    label = release.get("dataset_version", release.get("file", "archived"))
    expected_rows = release["row_count"]
    if len(rows) != expected_rows:
        errors.append(f"{label}: expected {expected_rows} rows, found {len(rows)}")

    keys = [(row["model"], row["question_id"], row["condition"]) for row in rows]
    if len(keys) != len(set(keys)):
        errors.append(f"{label}: duplicate (model, question_id, condition) rows")

    models = sorted(manifest["models"])
    if sorted({row["model"] for row in rows}) != models:
        errors.append(f"{label}: models {sorted({row['model'] for row in rows})} != {models}")

    questions_per_model = manifest["questions_per_model"]
    for model in models:
        for condition in CONDITION_NAMES:
            count = sum(
                1 for row in rows if row["model"] == model and row["condition"] == condition
            )
            if count != questions_per_model:
                errors.append(
                    f"{label}: {model} {condition}: expected {questions_per_model} rows, found {count}"
                )

    strict_errors = validate_rows(rows, question_ids)
    if release.get("passes_strict_validation"):
        errors.extend(f"{label}: {error}" for error in strict_errors)
    else:
        expected_count = release.get("error_count_at_freeze")
        if expected_count is not None and len(strict_errors) != expected_count:
            errors.append(
                f"{label}: expected {expected_count} documented strict-validation "
                f"message(s), found {len(strict_errors)}"
            )
        invalid_conf = release.get("invalid_confidence_zero_rows")
        if invalid_conf is not None:
            invalid_rows = sum(1 for row in rows if not is_valid_row(row))
            if invalid_rows != invalid_conf:
                errors.append(
                    f"{label}: expected {invalid_conf} schema-invalid row(s), found {invalid_rows}"
                )

    return errors


def validate_manifest_contract(
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    *,
    data_dir: Path | None = None,
) -> list[str]:
    """Validate frozen dataset artifacts against manifest.json."""
    manifest = load_manifest(manifest_path)
    data_dir = data_dir or manifest_path.parent
    question_ids = load_question_ids()
    errors: list[str] = []

    for release in manifest.get("archived_releases", []):
        path = data_dir / release["file"]
        if not path.exists():
            errors.append(f"archived file missing: {path}")
            continue
        rows = load_jsonl_file(path)
        errors.extend(validate_archived_release(rows, release, manifest, question_ids))

    validation = manifest.get("validation", {})
    primary_name = manifest.get("primary_file", "results_clean.jsonl")
    primary_path = data_dir / primary_name
    primary_present = primary_path.exists()

    if validation.get("primary_file_present") is False and not primary_present:
        pass
    elif primary_present:
        rows = load_jsonl_file(primary_path)
        if validation.get("passes_strict_validation"):
            row_count = manifest.get("row_count")
            if row_count is not None and len(rows) != row_count:
                errors.append(
                    f"primary file: expected {row_count} rows, found {len(rows)}"
                )
            errors.extend(
                f"primary file: {error}"
                for error in validate_rows(rows, question_ids)
            )
        else:
            errors.append(
                f"primary file {primary_path.name} exists but "
                "manifest.validation.passes_strict_validation is false"
            )
    elif validation.get("passes_strict_validation"):
        errors.append(f"primary file missing: {primary_path}")

    status = manifest.get("status")
    if status == "pending_repair":
        purged = manifest.get("rows_purged_awaiting_recollection")
        if purged is None:
            errors.append("pending_repair manifest missing rows_purged_awaiting_recollection")

    return errors


def report_strict_validation(
    rows: list[dict],
    question_ids: set[str],
    *,
    label: str,
    allow_duplicates: bool = False,
) -> int:
    clean_rows, dup_count = dedupe_rows(rows, keep="last")
    errors = validate_rows(clean_rows, question_ids)
    if dup_count and not allow_duplicates:
        errors.insert(0, f"found {dup_count} duplicate row(s)")

    if errors:
        print(f"Strict validation failed ({label}):", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    models = sorted({row["model"] for row in clean_rows})
    print(
        f"Strict validation OK ({label}): {len(clean_rows)} rows, "
        f"{len(models)} model(s), {EXPECTED_QUESTIONS} questions × {len(CONDITION_NAMES)} conditions"
    )
    return 0


def dedupe_rows(rows: list[dict], *, keep: str = "last") -> tuple[list[dict], int]:
    """Deduplicate by (model, question_id, condition). Returns rows and duplicate count."""
    if keep not in {"first", "last"}:
        raise ValueError("keep must be 'first' or 'last'")

    keyed: dict[tuple[str, str, str], dict] = {}
    order: list[tuple[str, str, str]] = []
    duplicates = 0

    for row in rows:
        key = (row["model"], row["question_id"], row["condition"])
        if key in keyed:
            duplicates += 1
            if keep == "last":
                keyed[key] = row
        else:
            order.append(key)
            keyed[key] = row

    return [keyed[key] for key in order], duplicates


def sort_rows(rows: list[dict]) -> list[dict]:
    condition_rank = {name: index for index, name in enumerate(CONDITION_NAMES)}

    def sort_key(row: dict) -> tuple:
        return (
            row["model"],
            row["question_id"],
            condition_rank.get(row["condition"], len(CONDITION_NAMES)),
        )

    return sorted(rows, key=sort_key)


def validate_rows(
    rows: list[dict],
    question_ids: set[str],
    *,
    expected_questions: int = EXPECTED_QUESTIONS,
) -> list[str]:
    errors: list[str] = []

    counts: dict[tuple[str, str], int] = Counter()
    duplicate_keys: list[tuple[str, str, str]] = []
    seen_keys: set[tuple[str, str, str]] = set()

    for index, row in enumerate(rows, start=1):
        model = row.get("model")
        question_id = row.get("question_id")
        condition = row.get("condition")

        if not model or not question_id or not condition:
            errors.append(f"line {index}: missing model, question_id, or condition")
            continue

        key = (model, question_id, condition)
        if key in seen_keys:
            duplicate_keys.append(key)
        seen_keys.add(key)
        counts[(model, condition)] += 1

        if question_id not in question_ids:
            errors.append(f"{model} {question_id} {condition}: unknown question_id")

        if condition not in CONDITION_NAMES:
            errors.append(f"{model} {question_id} {condition}: unknown condition")

        if not is_valid_answer(row.get("answer")):
            errors.append(
                f"{model} {question_id} {condition}: answer not yes/no ({row.get('answer')!r})"
            )

        if not is_valid_confidence(row.get("confidence")):
            errors.append(
                f"{model} {question_id} {condition}: confidence not in 1-10 ({row.get('confidence')!r})"
            )

        if not is_valid_answer(row.get("prior_answer")):
            errors.append(
                f"{model} {question_id} {condition}: prior_answer not yes/no ({row.get('prior_answer')!r})"
            )

        if not is_valid_confidence(row.get("prior_confidence")):
            errors.append(
                f"{model} {question_id} {condition}: prior_confidence not in 1-10 "
                f"({row.get('prior_confidence')!r})"
            )

    if duplicate_keys:
        preview = ", ".join(f"{m}/{q}/{c}" for m, q, c in duplicate_keys[:5])
        extra = "" if len(duplicate_keys) <= 5 else f" (+{len(duplicate_keys) - 5} more)"
        errors.append(f"duplicate (model, question_id, condition) rows: {preview}{extra}")

    models = sorted({row["model"] for row in rows if row.get("model")})
    for model in models:
        for condition in CONDITION_NAMES:
            count = counts[(model, condition)]
            if count != expected_questions:
                errors.append(
                    f"{model} {condition}: expected {expected_questions} rows, found {count}"
                )

        model_question_ids = {
            row["question_id"]
            for row in rows
            if row.get("model") == model and row.get("question_id")
        }
        missing = sorted(question_ids - model_question_ids)
        extra = sorted(model_question_ids - question_ids)
        if missing:
            errors.append(f"{model}: missing question_ids {', '.join(missing[:10])}" + (
                f" (+{len(missing) - 10} more)" if len(missing) > 10 else ""
            ))
        if extra:
            errors.append(f"{model}: unexpected question_ids {', '.join(extra[:10])}" + (
                f" (+{len(extra) - 10} more)" if len(extra) > 10 else ""
            ))

    return errors


def row_is_schema_valid(row: dict) -> bool:
    return is_valid_row(row)


def keys_to_purge(rows: list[dict]) -> set[tuple[str, str, str]]:
    """Return (model, question_id, condition) keys that fail schema validation."""
    purge: set[tuple[str, str, str]] = set()

    bad_prior_questions: set[tuple[str, str]] = set()
    for row in rows:
        if row.get("condition") != "prior":
            continue
        if not row_is_schema_valid(row):
            bad_prior_questions.add((row["model"], row["question_id"]))

    for row in rows:
        key = (row["model"], row["question_id"], row["condition"])
        if (row["model"], row["question_id"]) in bad_prior_questions:
            purge.add(key)
            continue
        if not row_is_schema_valid(row):
            purge.add(key)

    return purge


def purge_invalid_rows(rows: list[dict]) -> tuple[list[dict], int]:
    purge = keys_to_purge(rows)
    if not purge:
        return rows, 0
    kept = [
        row
        for row in rows
        if (row["model"], row["question_id"], row["condition"]) not in purge
    ]
    return kept, len(purge)


def purge_invalid_in_results_dir(results_dir: Path = RESULTS_DIR) -> dict[str, int]:
    """Remove invalid rows from each per-model JSONL file; returns purge counts."""
    paths = discover_result_files(results_dir, include_legacy=False)
    counts: dict[str, int] = {}
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        if not rows:
            continue
        model = rows[0]["model"]
        cleaned, removed = purge_invalid_rows(rows)
        if removed:
            write_jsonl(path, sort_rows(cleaned))
        counts[model] = removed
    return counts


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_rows_grouped_by_model(paths: list[Path]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                grouped[row["model"]].append(row)
    return grouped


def fix_per_model_files(results_dir: Path, paths: list[Path]) -> int:
    """Rewrite per-model JSONL files with duplicates removed (last row wins)."""
    grouped = load_rows_grouped_by_model(paths)
    removed = 0

    for model, rows in grouped.items():
        deduped, dup_count = dedupe_rows(rows, keep="last")
        removed += dup_count
        out_path = model_results_path(model, results_dir)
        write_jsonl(out_path, sort_rows(deduped))

    return removed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate persuasion experiment results.",
        epilog=(
            "Modes: default validates results/ (strict). "
            "--contract checks frozen artifacts per manifest.json. "
            "--input runs strict validation on one JSONL file. "
            "--write-clean writes a merged snapshot (output via --output)."
        ),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Directory containing per-model JSONL files (default mode)",
    )
    parser.add_argument(
        "--legacy-path",
        type=Path,
        default=LEGACY_RESULTS_PATH,
        help="Legacy combined JSONL fallback",
    )
    parser.add_argument(
        "--input",
        type=Path,
        metavar="PATH",
        help="Strict-validate a single merged JSONL file (read-only)",
    )
    parser.add_argument(
        "--contract",
        action="store_true",
        help="Validate frozen dataset contract from data/final/manifest.json",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help="Manifest path for --contract",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Deduplicate per-model JSONL files in place (last row wins)",
    )
    parser.add_argument(
        "--write-clean",
        action="store_true",
        help="Write deduplicated merged snapshot after strict validation passes",
    )
    parser.add_argument(
        "--purge-invalid",
        action="store_true",
        help="Remove schema-invalid rows from per-model JSONL files (for re-collection)",
    )
    parser.add_argument(
        "--output",
        "--clean-path",
        type=Path,
        default=DEFAULT_CLEAN_PATH,
        dest="output",
        help="Output path for --write-clean (default: data/final/results_clean.jsonl)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    question_ids = load_question_ids()

    if args.input and (args.contract or args.write_clean or args.fix or args.purge_invalid):
        print("--input cannot be combined with --contract, --write-clean, --fix, or --purge-invalid",
              file=sys.stderr)
        raise SystemExit(2)

    if args.contract:
        errors = validate_manifest_contract(args.manifest)
        if errors:
            print("Contract validation failed:", file=sys.stderr)
            for error in errors:
                print(f"  - {error}", file=sys.stderr)
            raise SystemExit(1)
        manifest = load_manifest(args.manifest)
        version = manifest.get("dataset_version", "unknown")
        print(f"Contract validation OK ({version})")
        raise SystemExit(0)

    if args.input:
        if not args.input.exists():
            print(f"Input file not found: {args.input}", file=sys.stderr)
            raise SystemExit(1)
        rows = load_jsonl_file(args.input)
        raise SystemExit(
            report_strict_validation(rows, question_ids, label=str(args.input))
        )

    paths = discover_result_files(args.results_dir, args.legacy_path, include_legacy=False)
    if not paths and args.legacy_path.exists():
        paths = [args.legacy_path]

    if not paths:
        print(f"No result files found under {args.results_dir}", file=sys.stderr)
        raise SystemExit(1)

    rows = load_jsonl_rows(paths)

    if args.fix:
        removed = fix_per_model_files(args.results_dir, paths)
        print(f"Deduplicated per-model files ({removed} duplicate row(s) removed).")
        paths = discover_result_files(args.results_dir, args.legacy_path, include_legacy=False)
        rows = load_jsonl_rows(paths)

    if args.purge_invalid:
        purged = purge_invalid_in_results_dir(args.results_dir)
        total = sum(purged.values())
        if total:
            print(f"Purged {total} invalid row(s) from results/:")
            for model, count in sorted(purged.items()):
                if count:
                    print(f"  {model}: {count}")
            print(
                "Re-run: python main.py --model meta/llama-3.1-8b-instruct\n"
                "Then:  python validate_results.py --write-clean"
            )
        else:
            print("No invalid rows to purge.")
        paths = discover_result_files(args.results_dir, args.legacy_path, include_legacy=False)
        rows = load_jsonl_rows(paths)

    if args.purge_invalid and not args.write_clean and not args.fix:
        clean_rows, _ = dedupe_rows(rows, keep="last")
        errors = validate_rows(clean_rows, question_ids)
        if errors:
            print(f"\nPost-purge validation: {len(errors)} issue(s) remain (expected until re-collection).")
            raise SystemExit(2)
        raise SystemExit(0)

    clean_rows, dup_count = dedupe_rows(rows, keep="last")
    clean_rows = sort_rows(clean_rows)

    if args.write_clean:
        errors = validate_rows(clean_rows, question_ids)
        if errors:
            print("Refusing to write clean snapshot — validation failed:", file=sys.stderr)
            for error in errors:
                print(f"  - {error}", file=sys.stderr)
            raise SystemExit(1)
        write_jsonl(args.output, clean_rows)
        print(f"Wrote {len(clean_rows)} rows to {args.output}")

    audit_fields = ("prompt", "raw_response")
    missing_audit = sum(
        1 for row in clean_rows if not row.get("prompt") or not row.get("raw_response")
    )
    if missing_audit:
        print(
            f"Audit note: {missing_audit}/{len(clean_rows)} rows lack prompt or raw_response "
            "(see data/final/README.md).",
            file=sys.stderr,
        )

    errors = validate_rows(clean_rows, question_ids)
    if dup_count and not args.fix and not args.write_clean:
        errors.insert(0, f"found {dup_count} duplicate row(s); run with --fix or --write-clean")

    if errors:
        print("Validation failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        raise SystemExit(1)

    models = sorted({row["model"] for row in clean_rows})
    print(
        f"Validation OK: {len(clean_rows)} rows, "
        f"{len(models)} model(s), {EXPECTED_QUESTIONS} questions × {len(CONDITION_NAMES)} conditions"
    )


if __name__ == "__main__":
    main()
