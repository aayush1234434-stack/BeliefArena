# Data Card — Belief Persuasion Benchmark

| Field | Value |
|-------|--------|
| **Dataset version** | `1.1.0` |
| **Status** | Frozen — validator-clean |
| **Schema version** | v1 (parsed answer/confidence only) |
| **Primary file** | [`results_clean.jsonl`](results_clean.jsonl) |
| **Archived v1.0.0** | [`results_clean_v1.0.0_invalid.jsonl`](results_clean_v1.0.0_invalid.jsonl) |
| **Machine manifest** | [`manifest.json`](manifest.json) |

## Contents

| Metric | Value |
|--------|------:|
| Total rows | 1,600 |
| Models | 2 |
| Questions per model | 100 |
| Conditions per model | 8 |
| Trials per model | 800 |

### Models

| Model ID | Rows |
|----------|-----:|
| `meta/llama-3.1-8b-instruct` | 800 |
| `qwen/qwen3.5-122b-a10b` | 800 |

### Conditions

`prior`, `strength_alone`, `majority_plain_alone`, `majority_vague_alone`, `credibility_alone`, `strength_vs_majority`, `majority_vs_credibility`, `strength_vs_credibility`

## Repair history (v1.0.0 → v1.1.0)

The v1.0.0 freeze (1,600 rows) **failed strict validation**: 41 `meta/llama-3.1-8b-instruct` rows reported `confidence` or `prior_confidence` as `"0"`. That file is archived as `results_clean_v1.0.0_invalid.jsonl`.

**v1.1.0 repair:**

1. Invalid rows were purged from `results/meta_llama_3_1_8b_instruct.jsonl`.
2. 41 Llama trials were re-collected via `main.py`.
3. `validate_results.py --write-clean` produced `results_clean.jsonl` (strict validation passes).

## Audit trail note

1,559 of 1,600 rows lack `prompt` / `raw_response` fields from the original v1 collection. The 41 re-collected Llama trials include these audit fields.

## Known limitations

1. **Partial audit trail** — Most rows lack `prompt` / `raw_response`.
2. **Single sample** — One API draw per trial at temperature 0.7.
3. **Small benchmark** — 100 binary questions.
4. **Two models** — NVIDIA-hosted Llama 3.1 8B Instruct and Qwen 3.5 122B A10B only.
5. **Synthetic persuasion texts** — Authored evidence templates, not organic misinformation.

## Usage

```bash
python validate_results.py --contract
python validate_results.py
python validate_results.py --input data/final/results_clean.jsonl
python -m unittest discover -s tests -v
```

## Changelog

- **1.1.0** (2026-07-02) — Repaired 41 invalid Llama trials; validator-clean freeze.
- **1.0.0** (2026-07-02) — Initial freeze; failed strict validation (archived).
