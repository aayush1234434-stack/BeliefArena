# Winner Analysis

## By competitive condition

| condition | label | strategy | wins | win_pct |
| --- | --- | --- | --- | --- |
| strength_vs_majority | Strength vs majority | strength | 100 | 100.00 |
| strength_vs_majority | Strength vs majority | majority | 0 | 0.00 |
| strength_vs_majority | Strength vs majority | credibility | 0 | 0.00 |
| majority_vs_credibility | Majority vs credibility | strength | 0 | 0.00 |
| majority_vs_credibility | Majority vs credibility | majority | 99 | 99.00 |
| majority_vs_credibility | Majority vs credibility | credibility | 1 | 1.00 |
| strength_vs_credibility | Strength vs credibility | strength | 99 | 99.00 |
| strength_vs_credibility | Strength vs credibility | majority | 0 | 0.00 |
| strength_vs_credibility | Strength vs credibility | credibility | 1 | 1.00 |

## Overall strategy ranking

| strategy | wins | win_pct | rank | ci_95_lo | ci_95_hi |
| --- | --- | --- | --- | --- | --- |
| strength | 199 | 66.33 | 1 | 60.81 | 71.44 |
| majority | 99 | 33.00 | 2 | 27.92 | 38.51 |
| credibility | 2 | 0.67 | 3 | 0.18 | 2.40 |

## Inferential tests (this sample)

Per condition: two-sided binomial test of whether the more frequent side differs from 50%; 95% Wilson confidence intervals (CI) for the leading side's win rate.

| condition | label | comparison | leading_strategy | leading_wins | n_trials | leading_win_pct | ci_95_lo | ci_95_hi | binom_p_value |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| strength_vs_majority | Strength vs majority | strength vs majority | strength | 100 | 100 | 100 | 96.3 | 100 | 1.578e-30 |
| majority_vs_credibility | Majority vs credibility | majority vs credibility | majority | 99 | 100 | 99 | 94.55 | 99.82 | 1.593e-28 |
| strength_vs_credibility | Strength vs credibility | strength vs credibility | strength | 99 | 100 | 99 | 94.55 | 99.82 | 1.593e-28 |

**Overall pooled distribution** (*n* = 300 competitive trials): strength 199, majority 99, credibility 2. Chi-square goodness-of-fit vs. uniform three-way split: χ² = 194.060, *p* = 7.251e-43. This tests equality of pooled win counts, not pairwise dominance in a single matchup.
