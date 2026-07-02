# Statistical Tests

| test | comparison | statistic | p_value | effect_size | effect_label | p_holm |
| --- | --- | --- | --- | --- | --- | --- |
| McNemar: prior vs strength_alone | strength_alone | b=0, c=1 | 1 |  | n/a | 1 |
| Wilcoxon: confidence prior vs strength_alone | strength_alone | signed-rank | 0.00729 | 0.2771 | Cohen's d | 0.1239 |
| McNemar: prior vs majority_plain_alone | majority_plain_alone | b=9, c=1 | 0.02148 |  | n/a | 0.3008 |
| Wilcoxon: confidence prior vs majority_plain_alone | majority_plain_alone | signed-rank | 0.002236 | -0.3286 | Cohen's d | 0.04025 |
| McNemar: prior vs majority_vague_alone | majority_vague_alone | b=6, c=1 | 0.125 |  | n/a | 1 |
| Wilcoxon: confidence prior vs majority_vague_alone | majority_vague_alone | signed-rank | 0.1244 | -0.2328 | Cohen's d | 1 |
| McNemar: prior vs credibility_alone | credibility_alone | b=0, c=1 | 1 |  | n/a | 1 |
| Wilcoxon: confidence prior vs credibility_alone | credibility_alone | signed-rank | 0.1306 | 0.1522 | Cohen's d | 1 |
| McNemar: prior vs strength_vs_majority | strength_vs_majority | b=0, c=1 | 1 |  | n/a | 1 |
| Wilcoxon: confidence prior vs strength_vs_majority | strength_vs_majority | signed-rank | 0.00729 | 0.2771 | Cohen's d | 0.1239 |
| McNemar: prior vs majority_vs_credibility | majority_vs_credibility | b=1, c=1 | 1 |  | n/a | 1 |
| Wilcoxon: confidence prior vs majority_vs_credibility | majority_vs_credibility | signed-rank | 2.452e-07 | -0.4316 | Cohen's d | 4.659e-06 |
| McNemar: prior vs strength_vs_credibility | strength_vs_credibility | b=1, c=1 | 1 |  | n/a | 1 |
| Wilcoxon: confidence prior vs strength_vs_credibility | strength_vs_credibility | signed-rank | 6.367e-09 | -0.5319 | Cohen's d | 1.273e-07 |
| Fisher: flip rate strength_alone vs majority_plain_alone | majority_plain_alone | 2x2 exact | 0.009658 |  | n/a | 0.1449 |
| Fisher: flip rate strength_alone vs majority_vague_alone | majority_vague_alone | 2x2 exact | 0.06486 |  | n/a | 0.8432 |
| Fisher: flip rate strength_alone vs credibility_alone | credibility_alone | 2x2 exact | 1 |  | n/a | 1 |
| Fisher: flip rate strength_alone vs strength_vs_majority | strength_vs_majority | 2x2 exact | 1 |  | n/a | 1 |
| Fisher: flip rate strength_alone vs majority_vs_credibility | majority_vs_credibility | 2x2 exact | 1 |  | n/a | 1 |
| Fisher: flip rate strength_alone vs strength_vs_credibility | strength_vs_credibility | 2x2 exact | 1 |  | n/a | 1 |

**Flip-rate omnibus test:** Some flip-count cells are sparse (<5); pairwise Fisher's exact tests reported below.
