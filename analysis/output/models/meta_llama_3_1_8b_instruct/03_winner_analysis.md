# Winner Analysis

## By competitive condition

| condition | label | strategy | wins | win_pct |
| --- | --- | --- | --- | --- |
| strength_vs_majority | Strength vs majority | strength | 99 | 99.00 |
| strength_vs_majority | Strength vs majority | majority | 1 | 1.00 |
| strength_vs_majority | Strength vs majority | credibility | 0 | 0.00 |
| majority_vs_credibility | Majority vs credibility | strength | 0 | 0.00 |
| majority_vs_credibility | Majority vs credibility | majority | 88 | 88.00 |
| majority_vs_credibility | Majority vs credibility | credibility | 12 | 12.00 |
| strength_vs_credibility | Strength vs credibility | strength | 82 | 82.00 |
| strength_vs_credibility | Strength vs credibility | majority | 0 | 0.00 |
| strength_vs_credibility | Strength vs credibility | credibility | 18 | 18.00 |

## Overall strategy ranking

| strategy | wins | win_pct | rank | ci_95_lo | ci_95_hi |
| --- | --- | --- | --- | --- | --- |
| strength | 181 | 60.33 | 1 | 54.70 | 65.70 |
| majority | 89 | 29.67 | 2 | 24.78 | 35.07 |
| credibility | 30 | 10.00 | 3 | 7.09 | 13.92 |

## Inferential tests (this sample)

Per condition: two-sided binomial test of whether the more frequent side differs from 50%; 95% Wilson confidence intervals (CI) for the leading side's win rate.

| condition | label | comparison | leading_strategy | leading_wins | n_trials | leading_win_pct | ci_95_lo | ci_95_hi | binom_p_value |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| strength_vs_majority | Strength vs majority | strength vs majority | strength | 99 | 100 | 99 | 94.55 | 99.82 | 1.593e-28 |
| majority_vs_credibility | Majority vs credibility | majority vs credibility | majority | 88 | 100 | 88 | 80.19 | 93 | 1.911e-15 |
| strength_vs_credibility | Strength vs credibility | strength vs credibility | strength | 82 | 100 | 82 | 73.33 | 88.3 | 6.148e-11 |

**Overall pooled distribution** (*n* = 300 competitive trials): strength 181, majority 89, credibility 30. Chi-square goodness-of-fit vs. uniform three-way split: χ² = 115.820, *p* = 7.08e-26. This tests equality of pooled win counts, not pairwise dominance in a single matchup.
