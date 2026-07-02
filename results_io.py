"""Helpers for per-model experiment result files under ``results/``."""

from __future__ import annotations

import json
import re
from pathlib import Path

RESULTS_DIR = Path("results")
LEGACY_RESULTS_PATH = Path("results.jsonl")
FROZEN_CLEAN_PATH = Path("data") / "final" / "results_clean.jsonl"
ARCHIVED_INVALID_CLEAN_PATH = Path("data") / "final" / "results_clean_v1.0.0_invalid.jsonl"


def slugify_model(model: str) -> str:
    """Convert a model id to a safe filename stem."""
    return re.sub(r"[^a-zA-Z0-9]+", "_", model).strip("_").lower()


def model_results_path(model: str, results_dir: Path = RESULTS_DIR) -> Path:
    """Return the JSONL path for a single model's results."""
    return results_dir / f"{slugify_model(model)}.jsonl"


def model_metadata_path(model: str, results_dir: Path = RESULTS_DIR) -> Path:
    """Return the JSON metadata path for a single model's run configuration."""
    return results_dir / f"{slugify_model(model)}.metadata.json"


def write_model_metadata(metadata: dict, model: str, results_dir: Path = RESULTS_DIR) -> Path:
    """Write run metadata for a model; returns the path written."""
    path = model_metadata_path(model, results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")
    return path


def migrate_legacy_results(
    legacy_path: Path = LEGACY_RESULTS_PATH,
    results_dir: Path = RESULTS_DIR,
    overwrite: bool = False,
) -> list[Path]:
    """
    Split a legacy ``results.jsonl`` into per-model files under ``results/``.

    By default, skips models whose destination file already exists.
    """
    if not legacy_path.exists():
        return []

    rows_by_model: dict[str, list[dict]] = {}
    with legacy_path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows_by_model.setdefault(row["model"], []).append(row)

    results_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for model, rows in rows_by_model.items():
        path = model_results_path(model, results_dir)
        if path.exists() and not overwrite:
            continue
        mode = "w" if overwrite or not path.exists() else "a"
        with path.open(mode) as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        written.append(path)
    return written


def load_jsonl_rows(paths: list[Path]) -> list[dict]:
    """Load and concatenate rows from one or more JSONL files."""
    rows: list[dict] = []
    for path in paths:
        if not path.exists():
            continue
        with path.open() as handle:
            for line in handle:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    row["source_file"] = path.name
                    rows.append(row)
    return rows


def discover_result_files(
    results_dir: Path = RESULTS_DIR,
    legacy_path: Path = LEGACY_RESULTS_PATH,
    include_legacy: bool = True,
) -> list[Path]:
    """Find result JSONL files. Prefer ``results/``; fall back to legacy file."""
    if results_dir.exists():
        paths = sorted(results_dir.glob("*.jsonl"))
        if paths:
            return paths
    if include_legacy and legacy_path.exists():
        return [legacy_path]
    return []


def load_dataset_rows(
    *,
    root: Path | None = None,
    results_dir: Path = RESULTS_DIR,
    legacy_path: Path = LEGACY_RESULTS_PATH,
    frozen_path: Path = FROZEN_CLEAN_PATH,
) -> list[dict]:
    """Load rows for analysis: frozen clean snapshot when present, else merged ``results/``."""
    root = root or Path(__file__).resolve().parent
    clean = root / frozen_path
    if clean.exists():
        return load_jsonl_rows([clean])
    paths = discover_result_files(root / results_dir, root / legacy_path, include_legacy=True)
    return load_jsonl_rows(paths)
