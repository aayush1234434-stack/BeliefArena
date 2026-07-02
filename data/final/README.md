# Frozen results snapshot

See **[DATA_CARD.md](DATA_CARD.md)** for the full dataset card.

## Current status (`1.1.0`)

- **`results_clean.jsonl`** — validator-clean freeze (1,600 rows; strict validation passes).
- **`results_clean_v1.0.0_invalid.jsonl`** — archived 1,600-row v1.0.0 freeze (41 invalid Llama confidences; superseded).
- **`results/`** — 1,600 valid rows (800 per model).

## Validate

```bash
# Frozen artifact contract (archived v1.0.0 + current manifest)
python validate_results.py --contract

# Strict validation on live results/
python validate_results.py

# Strict validation on the frozen snapshot
python validate_results.py --input data/final/results_clean.jsonl

python -m unittest discover -s tests -v
```

`--output` / `--clean-path` is the **write** destination for `--write-clean` only, not a validation input.
