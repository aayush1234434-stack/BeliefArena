# Results (meta/llama-3.1-8b-instruct)


## Belief updating under persuasion

In this sample, we measured belief change as the proportion of trials in which the model's yes/no answer differed from its prior response. Across persuasion conditions, flip rates ranged from 24.0% to 37.0%. The highest flip rates occurred for Majority vs credibility (37.0%, 95% CI [28.2, 46.8]%), Majority vague (alone) (31.0%, 95% CI [22.8, 40.6]%), Credibility (alone) (31.0%, 95% CI [22.8, 40.6]%).

## Competitive strategy selection (head-to-head trials)

In this sample, **strength** was the most frequently selected strategy in pooled competitive trials (60.3%, 95% CI [54.7, 65.7]%). A chi-square test against a uniform three-way split gave χ² = 115.82, *p* = 7.08e-26. Per-matchup binomial tests and CIs are in the winner analysis table.

## Effects on factual accuracy

In this sample, persuasion conditions changed mean accuracy by +19.86 percentage points relative to the prior baseline. The largest net factual benefit was observed for Strength vs majority (net gain +28.0%), while the smallest net gain was observed for Majority plain (alone) (+10.0%).

## Confidence and calibration

Mean confidence shifted by -1.79 points on average. The prior calibration gap (confidence when correct minus confidence when incorrect) was -2.32; under persuasion conditions the mean gap was -3.11, indicating increased overconfidence on errors relative to baseline.

## Model differences

Analysis included meta/llama-3.1-8b-instruct (flip rate 29.0%).

## Statistical significance

Paired McNemar tests compared each persuasion condition's accuracy against the prior; Wilcoxon signed-rank tests assessed paired confidence shifts. After Holm–Bonferroni correction, 10 Holm-corrected tests reached p<0.05.

## Summary

In this sample, persuasion altered model beliefs (mean flip rate 29.0%) with heterogeneous effects on accuracy and confidence. Competitive selection patterns and susceptibility differences are reported with uncertainty intervals in the accompanying tables.
