# Correctness Analysis

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

## Transition matrices

### Strength (alone)

|  | Post: Correct | Post: Wrong |
|---|---:|---:|
| Prior: Correct | 170 | 0 |
| Prior: Wrong | 28 | 2 |

### Majority plain (alone)

|  | Post: Correct | Post: Wrong |
|---|---:|---:|
| Prior: Correct | 154 | 16 |
| Prior: Wrong | 18 | 12 |

### Majority vague (alone)

|  | Post: Correct | Post: Wrong |
|---|---:|---:|
| Prior: Correct | 159 | 11 |
| Prior: Wrong | 27 | 3 |

### Credibility (alone)

|  | Post: Correct | Post: Wrong |
|---|---:|---:|
| Prior: Correct | 167 | 3 |
| Prior: Wrong | 29 | 1 |

### Strength vs majority

|  | Post: Correct | Post: Wrong |
|---|---:|---:|
| Prior: Correct | 170 | 0 |
| Prior: Wrong | 29 | 1 |

### Majority vs credibility

|  | Post: Correct | Post: Wrong |
|---|---:|---:|
| Prior: Correct | 159 | 11 |
| Prior: Wrong | 28 | 2 |

### Strength vs credibility

|  | Post: Correct | Post: Wrong |
|---|---:|---:|
| Prior: Correct | 162 | 8 |
| Prior: Wrong | 19 | 11 |

