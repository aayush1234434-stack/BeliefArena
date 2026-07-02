# Correctness Analysis

| condition | label | accuracy_pct | accuracy_change_from_prior_pct | C_to_C | C_to_W | W_to_C | W_to_W |
| --- | --- | --- | --- | --- | --- | --- | --- |
| prior | Prior | 99.00 | 0.00 | 0 | 0 | 0 | 0 |
| strength_alone | Strength (alone) | 100.00 | 1.00 | 99 | 0 | 1 | 0 |
| majority_plain_alone | Majority plain (alone) | 91.00 | -8.00 | 90 | 9 | 1 | 0 |
| majority_vague_alone | Majority vague (alone) | 94.00 | -5.00 | 93 | 6 | 1 | 0 |
| credibility_alone | Credibility (alone) | 100.00 | 1.00 | 99 | 0 | 1 | 0 |
| strength_vs_majority | Strength vs majority | 100.00 | 1.00 | 99 | 0 | 1 | 0 |
| majority_vs_credibility | Majority vs credibility | 99.00 | 0.00 | 98 | 1 | 1 | 0 |
| strength_vs_credibility | Strength vs credibility | 99.00 | 0.00 | 98 | 1 | 1 | 0 |

## Transition matrices

### Strength (alone)

|  | Post: Correct | Post: Wrong |
|---|---:|---:|
| Prior: Correct | 99 | 0 |
| Prior: Wrong | 1 | 0 |

### Majority plain (alone)

|  | Post: Correct | Post: Wrong |
|---|---:|---:|
| Prior: Correct | 90 | 9 |
| Prior: Wrong | 1 | 0 |

### Majority vague (alone)

|  | Post: Correct | Post: Wrong |
|---|---:|---:|
| Prior: Correct | 93 | 6 |
| Prior: Wrong | 1 | 0 |

### Credibility (alone)

|  | Post: Correct | Post: Wrong |
|---|---:|---:|
| Prior: Correct | 99 | 0 |
| Prior: Wrong | 1 | 0 |

### Strength vs majority

|  | Post: Correct | Post: Wrong |
|---|---:|---:|
| Prior: Correct | 99 | 0 |
| Prior: Wrong | 1 | 0 |

### Majority vs credibility

|  | Post: Correct | Post: Wrong |
|---|---:|---:|
| Prior: Correct | 98 | 1 |
| Prior: Wrong | 1 | 0 |

### Strength vs credibility

|  | Post: Correct | Post: Wrong |
|---|---:|---:|
| Prior: Correct | 98 | 1 |
| Prior: Wrong | 1 | 0 |

