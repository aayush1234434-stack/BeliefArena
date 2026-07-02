# Correctness Analysis

| condition | label | accuracy_pct | accuracy_change_from_prior_pct | C_to_C | C_to_W | W_to_C | W_to_W |
| --- | --- | --- | --- | --- | --- | --- | --- |
| prior | Prior | 71.00 | 0.00 | 0 | 0 | 0 | 0 |
| strength_alone | Strength (alone) | 98.00 | 27.00 | 71 | 0 | 27 | 2 |
| majority_plain_alone | Majority plain (alone) | 81.00 | 10.00 | 64 | 7 | 17 | 12 |
| majority_vague_alone | Majority vague (alone) | 92.00 | 21.00 | 66 | 5 | 26 | 3 |
| credibility_alone | Credibility (alone) | 96.00 | 25.00 | 68 | 3 | 28 | 1 |
| strength_vs_majority | Strength vs majority | 99.00 | 28.00 | 71 | 0 | 28 | 1 |
| majority_vs_credibility | Majority vs credibility | 88.00 | 17.00 | 61 | 10 | 27 | 2 |
| strength_vs_credibility | Strength vs credibility | 82.00 | 11.00 | 64 | 7 | 18 | 11 |

## Transition matrices

### Strength (alone)

|  | Post: Correct | Post: Wrong |
|---|---:|---:|
| Prior: Correct | 71 | 0 |
| Prior: Wrong | 27 | 2 |

### Majority plain (alone)

|  | Post: Correct | Post: Wrong |
|---|---:|---:|
| Prior: Correct | 64 | 7 |
| Prior: Wrong | 17 | 12 |

### Majority vague (alone)

|  | Post: Correct | Post: Wrong |
|---|---:|---:|
| Prior: Correct | 66 | 5 |
| Prior: Wrong | 26 | 3 |

### Credibility (alone)

|  | Post: Correct | Post: Wrong |
|---|---:|---:|
| Prior: Correct | 68 | 3 |
| Prior: Wrong | 28 | 1 |

### Strength vs majority

|  | Post: Correct | Post: Wrong |
|---|---:|---:|
| Prior: Correct | 71 | 0 |
| Prior: Wrong | 28 | 1 |

### Majority vs credibility

|  | Post: Correct | Post: Wrong |
|---|---:|---:|
| Prior: Correct | 61 | 10 |
| Prior: Wrong | 27 | 2 |

### Strength vs credibility

|  | Post: Correct | Post: Wrong |
|---|---:|---:|
| Prior: Correct | 64 | 7 |
| Prior: Wrong | 18 | 11 |

