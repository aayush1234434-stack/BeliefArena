# BeliefArena

**How LLMs resolve conflicting persuasion cues on factual yes/no questions.**

A controlled benchmark comparing **strong evidence**, **majority opinion**, and **source credibility** — alone and in head-to-head competition. Dataset v1.1.0: 1,600 trials · 2 models · 100 questions · 8 conditions.

> **Headline:** Majority cues produce the highest flip rates when shown **alone**, but **strong evidence wins** when cues conflict on the same question.

---

## Results

All tables below pool both models (`meta/llama-3.1-8b-instruct`, `qwen/qwen3.5-122b-a10b`). *n* = 200 trials per condition (100 questions × 2 models). Full per-model tables: [`analysis/output/`](analysis/output/).

### Flip analysis

Belief change = answer differs from the prior (no-evidence) response.

| condition | label | n_questions | n_flipped | flip_rate_pct |
| --- | --- | --- | --- | --- |
| majority_vs_credibility | Majority vs credibility | 200 | 39 | 19.50 |
| majority_vague_alone | Majority vague (alone) | 200 | 38 | 19.00 |
| majority_plain_alone | Majority plain (alone) | 200 | 34 | 17.00 |
| credibility_alone | Credibility (alone) | 200 | 32 | 16.00 |
| strength_vs_majority | Strength vs majority | 200 | 29 | 14.50 |
| strength_alone | Strength (alone) | 200 | 28 | 14.00 |
| strength_vs_credibility | Strength vs credibility | 200 | 27 | 13.50 |
| prior | Prior | 200 | 0 | 0.00 |

**Mean flip rate across persuasion conditions: 16.2%**

---

### Winner analysis — by competitive condition

| condition | label | strategy | wins | win_pct |
| --- | --- | --- | --- | --- |
| strength_vs_majority | Strength vs majority | strength | 199 | 99.50 |
| strength_vs_majority | Strength vs majority | majority | 1 | 0.50 |
| strength_vs_majority | Strength vs majority | credibility | 0 | 0.00 |
| majority_vs_credibility | Majority vs credibility | strength | 0 | 0.00 |
| majority_vs_credibility | Majority vs credibility | majority | 187 | 93.50 |
| majority_vs_credibility | Majority vs credibility | credibility | 13 | 6.50 |
| strength_vs_credibility | Strength vs credibility | strength | 181 | 90.50 |
| strength_vs_credibility | Strength vs credibility | majority | 0 | 0.00 |
| strength_vs_credibility | Strength vs credibility | credibility | 19 | 9.50 |

### Winner analysis — overall strategy ranking

| strategy | wins | win_pct | rank | ci_95_lo | ci_95_hi |
| --- | --- | --- | --- | --- | --- |
| strength | 380 | 63.33 | 1 | 59.40 | 67.09 |
| majority | 188 | 31.33 | 2 | 27.75 | 35.15 |
| credibility | 32 | 5.33 | 3 | 3.80 | 7.43 |

### Winner analysis — inferential tests

| condition | label | leading_strategy | leading_wins | n_trials | leading_win_pct | ci_95_lo | ci_95_hi | binom_p_value |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| strength_vs_majority | Strength vs majority | strength | 199 | 200 | 99.5 | 97.22 | 99.91 | 2.502e-58 |
| majority_vs_credibility | Majority vs credibility | majority | 187 | 200 | 93.5 | 89.2 | 96.16 | 1.18e-40 |
| strength_vs_credibility | Strength vs credibility | strength | 181 | 200 | 90.5 | 85.64 | 93.83 | 2.476e-34 |

Pooled competitive trials (*n* = 600): χ² = 303.84, *p* = 1.052e-66 vs uniform three-way split.

---

### Correctness analysis

| condition | label | accuracy_pct | accuracy_change_from_prior_pct | C_to_C | C_to_W | W_to_C | W_to_W |
| --- | --- | --- | --- | --- | --- | --- | --- |
| prior | Prior | 85.00 | 0.00 | 0 | 0 | 0 | 0 |
| strength_alone | Strength (alone) | 99.00 | 14.00 | 170 | 0 | 28 | 2 |
| majority_plain_alone | Majority plain (alone) | 86.00 | 1.00 | 154 | 16 | 18 | 12 |
| majority_vague_alone | Majority vague (alone) | 93.00 | 8.00 | 159 | 11 | 27 | 3 |
| credibility_alone | Credibility (alone) | 98.00 | 13.00 | 167 | 3 | 29 | 1 |
| strength_vs_majority | Strength vs majority | 99.50 | 14.50 | 170 | 0 | 29 | 1 |
| majority_vs_credibility | Majority vs credibility | 93.50 | 8.50 | 159 | 11 | 28 | 2 |
| strength_vs_credibility | Strength vs credibility | 90.50 | 5.50 | 162 | 8 | 19 | 11 |

Transition matrices (C→C, C→W, W→C, W→W): [`analysis/output/aggregated/04_correctness_analysis.md`](analysis/output/aggregated/04_correctness_analysis.md)

---

### Persuasion benefit analysis

| condition | label | harmful_rate_pct | helpful_rate_pct | net_persuasion_gain_pct | harmful_n | helpful_n |
| --- | --- | --- | --- | --- | --- | --- |
| strength_vs_majority | Strength vs majority | 0.00 | 14.50 | 14.50 | 0 | 29 |
| strength_alone | Strength (alone) | 0.00 | 14.00 | 14.00 | 0 | 28 |
| credibility_alone | Credibility (alone) | 1.50 | 14.50 | 13.00 | 3 | 29 |
| majority_vs_credibility | Majority vs credibility | 5.50 | 14.00 | 8.50 | 11 | 28 |
| majority_vague_alone | Majority vague (alone) | 5.50 | 13.50 | 8.00 | 11 | 27 |
| strength_vs_credibility | Strength vs credibility | 4.00 | 9.50 | 5.50 | 8 | 19 |
| majority_plain_alone | Majority plain (alone) | 8.00 | 9.00 | 1.00 | 16 | 18 |

**Mean net persuasion gain: +9.21 percentage points**

---

### Confidence analysis

| condition | label | mean_confidence | median_confidence | std_confidence | mean_confidence_change | n_increased | n_decreased | n_unchanged |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| prior | Prior | 8.110 | 9.000 | 2.389 |  | 0 | 0 | 0 |
| strength_alone | Strength (alone) | 8.265 | 10.000 | 2.803 | 0.155 | 67 | 40 | 93 |
| majority_plain_alone | Majority plain (alone) | 6.440 | 8.000 | 3.579 | -1.670 | 19 | 93 | 88 |
| majority_vague_alone | Majority vague (alone) | 6.595 | 8.000 | 3.550 | -1.515 | 22 | 85 | 93 |
| credibility_alone | Credibility (alone) | 8.465 | 9.000 | 2.437 | 0.355 | 68 | 39 | 93 |
| strength_vs_majority | Strength vs majority | 6.680 | 9.000 | 3.729 | -1.430 | 28 | 79 | 93 |
| majority_vs_credibility | Majority vs credibility | 6.135 | 9.000 | 3.620 | -1.975 | 21 | 125 | 54 |
| strength_vs_credibility | Strength vs credibility | 6.850 | 8.000 | 3.092 | -1.260 | 19 | 101 | 80 |

**Mean confidence change: −1.05 points** — persuasion shifts answers more than confidence.

---

### Confidence calibration

| condition | label | mean_confidence_when_correct | mean_confidence_when_incorrect | calibration_gap |
| --- | --- | --- | --- | --- |
| prior | Prior | 8.094 | 8.200 | -0.106 |
| strength_alone | Strength (alone) | 8.253 | 9.500 | -1.247 |
| majority_plain_alone | Majority plain (alone) | 6.634 | 5.250 | 1.384 |
| majority_vague_alone | Majority vague (alone) | 6.624 | 6.214 | 0.409 |
| credibility_alone | Credibility (alone) | 8.449 | 9.250 | -0.801 |
| strength_vs_majority | Strength vs majority | 6.668 | 9.000 | -2.332 |
| majority_vs_credibility | Majority vs credibility | 6.241 | 4.615 | 1.625 |
| strength_vs_credibility | Strength vs credibility | 6.807 | 7.263 | -0.457 |

---

### Model comparison

Ranked most resistant → most susceptible.

| rank | model | flip_rate_pct | accuracy_pct | mean_confidence | strength_win_pct | majority_win_pct | credibility_win_pct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | qwen/qwen3.5-122b-a10b | 3.43 | 97.75 | 9.42 | 66.33 | 33.00 | 0.67 |
| 2 | meta/llama-3.1-8b-instruct | 29.00 | 88.38 | 4.70 | 60.33 | 29.67 | 10.00 |

---

### Statistical tests

McNemar (accuracy vs prior) and Wilcoxon (confidence vs prior); Holm–Bonferroni corrected *p* in `p_holm`.

| test | comparison | statistic | p_value | effect_size | p_holm |
| --- | --- | --- | --- | --- | --- |
| McNemar: prior vs strength_alone | strength_alone | b=0, c=28 | 7.451e-09 |  | 5.96e-08 |
| McNemar: prior vs credibility_alone | credibility_alone | b=3, c=29 | 2.556e-06 |  | 1.789e-05 |
| McNemar: prior vs strength_vs_majority | strength_vs_majority | b=0, c=29 | 3.725e-09 |  | 3.353e-08 |
| McNemar: prior vs majority_plain_alone | majority_plain_alone | b=16, c=18 | 0.8642 |  | 0.8642 |
| McNemar: prior vs majority_vague_alone | majority_vague_alone | b=11, c=27 | 0.01385 |  | 0.06926 |
| McNemar: prior vs majority_vs_credibility | majority_vs_credibility | b=11, c=28 | 0.009475 |  | 0.05685 |
| McNemar: prior vs strength_vs_credibility | strength_vs_credibility | b=8, c=19 | 0.05224 |  | 0.1567 |
| Wilcoxon: confidence prior vs majority_plain_alone | majority_plain_alone | signed-rank | 1.814e-14 | Cohen's d = -0.598 | 2.358e-13 |
| Wilcoxon: confidence prior vs majority_vs_credibility | majority_vs_credibility | signed-rank | 1.3e-17 | Cohen's d = -0.701 | 1.82e-16 |

Flip-rate omnibus: χ² = 5.07, *p* = 0.535 (no significant difference in flip counts across conditions).

Mixed-effects regression tables: [`analysis/output/paper/`](analysis/output/paper/)

---

## Research questions

| # | Question | Answer |
|---|----------|--------|
| RQ1 | Which strategy changes beliefs most? | Majority cues (17–19.5% flip) |
| RQ2 | Which wins when strategies conflict? | **Strength** (63.3% pooled) |
| RQ3 | Does persuasion improve accuracy? | **Yes**, +9.21 pp mean net gain |
| RQ4 | Flips vs confidence-only change? | **Answer flips** dominate |
| RQ5 | Model susceptibility differences? | **Large** — Qwen 3.4% vs Llama 29% flip |
| RQ6 | Credibility vs majority? | **Majority** wins (93.5%) |
| RQ7 | Strength vs majority (head-to-head)? | **Strength** wins (99.5%) |

---

## Experiment design

- **100** binary factual questions across 62 topics ([`question.json`](question.json))
- **3 persuasion channels:** strength (detailed paragraph), majority (10 forum snippets), credibility (attributed sentence)
- **8 conditions:** prior + 4 alone + 3 head-to-head competitive
- **2 models** via [NVIDIA NIM API](https://build.nvidia.com/): Llama 3.1 8B Instruct, Qwen 3.5 122B A10B
- Temperature 0.7 · single sample per trial · 30 RPM pacing

| Condition | Evidence |
|-----------|----------|
| `prior` | None |
| `strength_alone` | Strong correct evidence |
| `majority_plain_alone` | Majority plain (wrong) |
| `majority_vague_alone` | Majority vague (wrong) |
| `credibility_alone` | Credible-source correct evidence |
| `strength_vs_majority` | Strong correct vs majority vague (wrong) |
| `majority_vs_credibility` | Majority plain (correct) vs credibility (wrong) |
| `strength_vs_credibility` | Strong correct vs credibility (wrong) |

Methods and limitations: [`paper/methods_and_limitations.md`](paper/methods_and_limitations.md)

---

## Dataset

| Field | Value |
|-------|-------|
| Version | **1.1.0** (frozen) |
| File | [`data/final/results_clean.jsonl`](data/final/results_clean.jsonl) |
| Rows | 1,600 |
| Manifest | [`data/final/manifest.json`](data/final/manifest.json) |

Do **not** use `results_clean_v1.0.0_invalid.jsonl` for analysis (archived provenance only).

---

## Reproducibility

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Validate
python validate_results.py --contract
python validate_results.py --input data/final/results_clean.jsonl
python -m unittest discover -s tests -v

# Regenerate all tables above
bash analysis/run_analysis.sh
python analysis/mixed_effects.py
```



## Project structure

```
BeliefArena/
├── README.md
├── LICENSE
├── requirements.txt
├── question.json                  # 100 questions + evidence (strength/majority/credibility)
├── main.py                        # Data collection (NVIDIA NIM API calls)
├── validate_results.py            # Schema validation + dataset freeze/contract check
│
├── data/
│   └── final/
│       ├── results_clean.jsonl              # Frozen dataset v1.1.0 (1,600 rows)
│       ├── results_clean_v1.0.0_invalid.jsonl  # Archived, do not use for analysis
│       └── manifest.json
│
├── analysis/
│   ├── run_analysis.sh            # Regenerates all result tables
│   ├── mixed_effects.py           # Mixed-effects regression models
│   └── output/
│       ├── aggregated/            # Pooled-model tables (flip, correctness, etc.)
│       ├── paper/                 # Regression tables for publication
│       └── 09_research_questions.md
│
├── paper/
│   └── methods_and_limitations.md
│
└── tests/
    └── ...                        # unittest suite (run via `python -m unittest discover`)

```
