# Belief Update Under Persuasion

Study how large language models update yes/no factual beliefs when exposed to different persuasion strategies (strength of evidence, majority opinion, credibility cues), alone and in head-to-head competition.

## Research questions

1. **RQ1** — Which persuasion strategy changes model beliefs most frequently?
2. **RQ2** — When strategies conflict, which is selected most often (with binomial tests)?
3. **RQ3** — Does persuasion improve or reduce factual accuracy?
4. **RQ4** — Does persuasion mainly change answers or only confidence?
5. **RQ5** — Are some models more resistant to persuasion than others?
6. **RQ6** — Does credibility outperform majority opinion?
7. **RQ7** — Does strong evidence consistently outperform majority?

Answers are generated in `analysis/output/09_research_questions.md` after running analysis.

## Experiment design

- **Stimuli:** 100 binary (Yes/No) factual questions in `question.json`, spanning common misconceptions and science topics.
- **Procedure:** For each question, the model first answers without persuasion (**prior**), then answers again under each persuasion **condition**. Prior responses are reused as the baseline for flip-rate and confidence-change calculations.
- **Response format:** The model must reply with two tokens: answer (`Yes` or `No`) and confidence (`1`–`10`), e.g. `No 7`.
- **Outcomes recorded per trial:** prompt, raw model response, parsed answer/confidence, prior answer/confidence, whether the answer **flipped** vs. prior, and (for competitive conditions) which strategy **won**.
- **Scale:** 100 questions × 8 conditions = **800 trials per model** (~700 API calls per model because the prior is asked once per question).

## Conditions

| Condition | Evidence shown |
|-----------|----------------|
| `prior` | None (baseline) |
| `strength_alone` | Strong correct evidence |
| `majority_plain_alone` | Majority plain (wrong) |
| `majority_vague_alone` | Majority vague (wrong) |
| `credibility_alone` | Credible-source correct evidence |
| `strength_vs_majority` | Strong correct vs. majority vague (wrong) |
| `majority_vs_credibility` | Majority plain (correct) vs. credibility (wrong) |
| `strength_vs_credibility` | Strong correct vs. credibility (wrong) |

Competitive conditions assign a **winner** (`strength`, `majority`, or `credibility`) based on whether the model's final answer matches the ground-truth correct answer.

## Models

Runs target models hosted on the [NVIDIA NIM API](https://build.nvidia.com/). Examples used in this repo:

| Model ID | Results file |
|----------|----------------|
| `meta/llama-3.1-8b-instruct` | `results/meta_llama_3_1_8b_instruct.jsonl` |
| `qwen/qwen3.5-122b-a10b` | `results/qwen_qwen3_5_122b_a10b.jsonl` |

Pass any NVIDIA API model id via `--model`. Each run writes a companion metadata file (see below).

### Run metadata

Each collection run writes `results/<model_slug>.metadata.json` with:

```json
{
  "model": "meta/llama-3.1-8b-instruct",
  "provider": "nvidia",
  "api_endpoint": "https://integrate.api.nvidia.com/v1",
  "temperature": 0.7,
  "max_tokens": 64,
  "prompt_version": "v1",
  "rpm_limit": 30,
  "run_started_at": "2026-07-01T12:00:00-07:00",
  "run_completed_at": "2026-07-01T14:30:00-07:00"
}
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export NVIDIA_API_KEY='your-nvidia-api-key'
```

`NVIDIA_API_KEY` is **required**; the script will exit if it is not set.

## Run data collection

```bash
# Default model (Llama 3.1 8B Instruct)
python main.py

# Specific model
python main.py --model qwen/qwen3.5-122b-a10b

# Custom results directory
python main.py --model meta/llama-3.1-8b-instruct --results-dir results

# Split legacy results.jsonl into per-model files
python main.py --migrate-legacy
```

**Notes:**

- Progress is checkpointed per question/condition; interrupted runs resume automatically.
- Requests are paced at 30 RPM to respect NVIDIA rate limits.
- Results append to `results/<model_slug>.jsonl` (one JSON object per line).
- New runs store `prompt`, `system_prompt`, and `raw_response` per trial. The frozen paper dataset (`data/final/`) was collected without these fields; see `data/final/README.md`.

## Run analysis

```bash
bash analysis/run_analysis.sh
# or
python analysis/persuasion_analysis.py
```

Analysis loads `data/final/results_clean.jsonl` when present, otherwise merged `results/*.jsonl`, and produces aggregated, per-model, and cross-model reports.

## Validate results

Two validators serve different purposes:

| Command | What it checks | Expected now |
|---------|----------------|--------------|
| `python validate_results.py --contract` | Frozen artifacts match `data/final/manifest.json` (shape, dedupe, documented strict-error count) | **Exit 0** |
| `python validate_results.py` | Strict schema on merged `results/` | **Exit 0** when complete |
| `python validate_results.py --input PATH` | Strict schema on one merged JSONL file (read-only) | Fails on archived v1.0.0 |

```bash
# Frozen dataset contract (archived v1.0.0 + pending manifest)
python validate_results.py --contract

# Strict validation on live collection files (fails until repair completes)
python validate_results.py

# Strict validation on a specific merged JSONL
python validate_results.py --input data/final/results_clean_v1.0.0_invalid.jsonl

# Deduplicate per-model JSONL files in place
python validate_results.py --fix

# Write frozen snapshot after strict validation passes (output path via --output)
python validate_results.py --write-clean
python validate_results.py --write-clean --output data/final/results_clean.jsonl

# Fix source files and freeze snapshot in one step
python validate_results.py --fix --write-clean

# Repair invalid Llama trials and freeze (requires NVIDIA_API_KEY)
bash scripts/repair_and_freeze.sh
```

`--write-clean` **refuses** to write unless strict validation passes. The archived v1.0.0 freeze intentionally fails strict checks (41 Llama rows with confidence `0`); use `--contract` to verify that artifact, not bare `validate_results.py`.

Strict validation checks:

- exactly 100 questions per model per condition (800 rows per model)
- no duplicate `(model, question_id, condition)` rows
- `answer` and `prior_answer` normalize to yes/no
- `confidence` and `prior_confidence` parse to 1–10
- all `question_id` values exist in `question.json`

## Run tests

```bash
python -m unittest discover -s tests -v
```

Covers answer/confidence parsing, deduplication, validation, outcome computation, and frozen dataset contract (`data/final/manifest.json`).

## Expected outputs

### Collection (`results/`)

| Path | Description |
|------|-------------|
| `results/<model_slug>.jsonl` | Trial-level results for one model |
| `results/<model_slug>.metadata.json` | Run configuration and timestamps |
| `data/final/results_clean.jsonl` | Validator-clean frozen snapshot (written only after `validate_results.py --write-clean` passes) |
| `data/final/results_clean_v1.0.0_invalid.jsonl` | Archived v1.0.0 freeze (41 invalid Llama confidences; not schema-clean) |
| `data/final/DATA_CARD.md` | Dataset version, exclusions, and known limitations |
| `data/final/manifest.json` | Machine-readable freeze metadata |
| `data/final/README.md` | Pointer to data card and validation commands |
| `results.jsonl` | Legacy combined file (optional; migrated automatically) |

### Analysis (`analysis/output/`)

| Path | Description |
|------|-------------|
| `README.md` | Index of all analysis outputs |
| `RESULTS.md` | Paper-style summary (aggregated) |
| `09_research_questions.md` | RQ1–RQ7 answers |
| `aggregated/` | Pooled analysis across all models (tables + `figures/`) |
| `models/<model_slug>/` | Per-model flip, confidence, winner, correctness, benefit, calibration, stats, figures |
| `model_comparison/` | Cross-model comparison tables and plots |

Per-model folders mirror the aggregated structure, e.g.:

- `01_flip_analysis.md`
- `02_confidence_analysis.md`
- `03_winner_analysis.md`
- `04_correctness_analysis.md`
- `05_persuasion_benefit.md`
- `06_calibration.md`
- `07_statistical_tests.md`
- `RESULTS.md`
- `figures/*.png`

## Paper

- [`paper/methods_and_limitations.md`](paper/methods_and_limitations.md) — Methods, Limitations, and tables for publication
- Regenerate regression tables: `python analysis/mixed_effects.py` (main effects + condition × model interactions)

## Project layout

```
belief/
├── main.py                    # Data collection
├── question.json              # Questions + evidence texts
├── question_texts.py          # Question text helpers
├── results_io.py              # Per-model result/metadata paths
├── results_schema.py            # Prompt/response schema helpers
├── validate_results.py          # Result validation and clean snapshot
├── results/                   # JSONL + metadata per model
├── data/final/                # Frozen clean dataset snapshot
├── analysis/
│   ├── persuasion_analysis.py # Analysis pipeline
│   ├── mixed_effects.py       # Paper regression tables
│   ├── run_analysis.sh
│   └── output/                # Generated reports (git-tracked or regenerate)
├── paper/
│   └── methods_and_limitations.md
├── requirements.txt
└── README.md
```
