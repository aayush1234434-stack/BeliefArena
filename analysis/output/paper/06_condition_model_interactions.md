# Condition × Model Interaction Tests

Models include all condition and model main effects plus condition × model interactions.
Reference condition: prior (confidence, accuracy) or majority plain alone (belief flip).
Reference model: Llama 3.1 8B Instruct (`qwen` = 0).

## Omnibus Wald tests (interaction block)

| Outcome | df | Statistic | *p* |
|---------|---:|----------:|----:|
| confidence | 7 | 155.658 | 2.628e-30 |
| accuracy | 7 | 57.062 | 5.809e-10 |
| belief_flip | 6 | 25.534 | 0.0002718 |

## Interaction coefficients

### confidence

| Term | Estimate | SE | *z*/*t* | *p* | 95% CI |
|------|----------|-----|---------|-----|--------|
| Strength (alone) × Qwen | -0.0700 | 0.4153 | -0.17 | 0.8661 | [-0.884, 0.744] |
| Majority plain (alone) × Qwen | 1.8000 | 0.4153 | 4.33 | &lt; .0001 | [0.986, 2.614] |
| Majority vague (alone) × Qwen | 2.0700 | 0.4153 | 4.98 | &lt; .0001 | [1.256, 2.884] |
| Credibility (alone) × Qwen | -0.5500 | 0.4153 | -1.32 | 0.1854 | [-1.364, 0.264] |
| Strength vs majority × Qwen | 3.1000 | 0.4153 | 7.46 | &lt; .0001 | [2.286, 3.914] |
| Majority vs credibility × Qwen | 2.8300 | 0.4153 | 6.81 | &lt; .0001 | [2.016, 3.644] |
| Strength vs credibility × Qwen | 1.2000 | 0.4153 | 2.89 | 0.003858 | [0.386, 2.014] |

### accuracy

| Term | Estimate | SE | *z*/*t* | *p* | 95% CI |
|------|----------|-----|---------|-----|--------|
| Strength (alone) × Qwen | -0.2600 | 0.0465 | -5.60 | &lt; .0001 | [-0.351, -0.169] |
| Majority plain (alone) × Qwen | -0.1800 | 0.0465 | -3.87 | 0.000107 | [-0.271, -0.089] |
| Majority vague (alone) × Qwen | -0.2600 | 0.0465 | -5.60 | &lt; .0001 | [-0.351, -0.169] |
| Credibility (alone) × Qwen | -0.2400 | 0.0465 | -5.17 | &lt; .0001 | [-0.331, -0.149] |
| Strength vs majority × Qwen | -0.2700 | 0.0465 | -5.81 | &lt; .0001 | [-0.361, -0.179] |
| Majority vs credibility × Qwen | -0.1700 | 0.0465 | -3.66 | 0.0002532 | [-0.261, -0.079] |
| Strength vs credibility × Qwen | -0.1100 | 0.0465 | -2.37 | 0.0179 | [-0.201, -0.019] |

### belief_flip

| Term | Estimate | SE | *z*/*t* | *p* | 95% CI |
|------|----------|-----|---------|-----|--------|
| Strength (alone) × Qwen | -2.5560 | 0.9186 | -2.78 | 0.005397 | [-4.356, -0.755] |
| Majority vague (alone) × Qwen | -0.7420 | 0.3879 | -1.91 | 0.05579 | [-1.502, 0.018] |
| Credibility (alone) × Qwen | -2.7505 | 0.9252 | -2.97 | 0.002951 | [-4.564, -0.937] |
| Strength vs majority × Qwen | -2.6061 | 0.9205 | -2.83 | 0.004636 | [-4.410, -0.802] |
| Majority vs credibility × Qwen | -2.3151 | 0.7496 | -3.09 | 0.002013 | [-3.784, -0.846] |
| Strength vs credibility × Qwen | -1.7487 | 0.7414 | -2.36 | 0.01835 | [-3.202, -0.295] |
