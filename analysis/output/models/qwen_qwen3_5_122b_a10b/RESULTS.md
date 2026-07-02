# Results (qwen/qwen3.5-122b-a10b)


## Belief updating under persuasion

In this sample, we measured belief change as the proportion of trials in which the model's yes/no answer differed from its prior response. Across persuasion conditions, flip rates ranged from 1.0% to 10.0%. The highest flip rates occurred for Majority plain (alone) (10.0%, 95% CI [5.5, 17.4]%), Majority vague (alone) (7.0%, 95% CI [3.4, 13.7]%), Majority vs credibility (2.0%, 95% CI [0.6, 7.0]%).

## Competitive strategy selection (head-to-head trials)

In this sample, **strength** was the most frequently selected strategy in pooled competitive trials (66.3%, 95% CI [60.8, 71.4]%). A chi-square test against a uniform three-way split gave χ² = 194.06, *p* = 7.251e-43. Per-matchup binomial tests and CIs are in the winner analysis table.

## Effects on factual accuracy

In this sample, persuasion conditions changed mean accuracy by -1.43 percentage points relative to the prior baseline. The largest net factual benefit was observed for Strength (alone) (net gain +1.0%), while the largest net harm was associated with Majority plain (alone) (-8.0%).

## Confidence and calibration

Mean confidence shifted by -0.31 points on average. The prior calibration gap (confidence when correct minus confidence when incorrect) was -0.27; under persuasion conditions the mean gap was 5.14, indicating reduced overconfidence on errors relative to baseline.

## Model differences

Analysis included qwen/qwen3.5-122b-a10b (flip rate 3.4%).

## Statistical significance

Paired McNemar tests compared each persuasion condition's accuracy against the prior; Wilcoxon signed-rank tests assessed paired confidence shifts. After Holm–Bonferroni correction, 3 Holm-corrected tests reached p<0.05.

## Summary

In this sample, persuasion altered model beliefs (mean flip rate 3.4%) with heterogeneous effects on accuracy and confidence. Competitive selection patterns and susceptibility differences are reported with uncertainty intervals in the accompanying tables.
