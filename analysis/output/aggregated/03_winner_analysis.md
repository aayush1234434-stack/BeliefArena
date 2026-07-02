# Winner Analysis

## By competitive condition

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

## Overall strategy ranking

| strategy | wins | win_pct | rank | ci_95_lo | ci_95_hi |
| --- | --- | --- | --- | --- | --- |
| strength | 380 | 63.33 | 1 | 59.40 | 67.09 |
| majority | 188 | 31.33 | 2 | 27.75 | 35.15 |
| credibility | 32 | 5.33 | 3 | 3.80 | 7.43 |

## Inferential tests (this sample)

Per condition: two-sided binomial test of whether the more frequent side differs from 50%; 95% Wilson confidence intervals (CI) for the leading side's win rate.

| condition | label | comparison | leading_strategy | leading_wins | n_trials | leading_win_pct | ci_95_lo | ci_95_hi | binom_p_value |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| strength_vs_majority | Strength vs majority | strength vs majority | strength | 199 | 200 | 99.5 | 97.22 | 99.91 | 2.502e-58 |
| majority_vs_credibility | Majority vs credibility | majority vs credibility | majority | 187 | 200 | 93.5 | 89.2 | 96.16 | 1.18e-40 |
| strength_vs_credibility | Strength vs credibility | strength vs credibility | strength | 181 | 200 | 90.5 | 85.64 | 93.83 | 2.476e-34 |

**Overall pooled distribution** (*n* = 600 competitive trials): strength 380, majority 188, credibility 32. Chi-square goodness-of-fit vs. uniform three-way split: χ² = 303.840, *p* = 1.052e-66. This tests equality of pooled win counts, not pairwise dominance in a single matchup.
