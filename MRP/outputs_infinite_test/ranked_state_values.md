# Ranked State Values

Each gamma block is sorted by decreasing value.

## Exact

### gamma = 0.00

| Rank | State | Value |
| --- | --- | ---: |
| 1 | Computer | 2.800000 |
| 2 | Coffee | 2.100000 |
| 3 | Home | 1.000000 |
| 4 | Chat | 0.300000 |

### gamma = 0.25

| Rank | State | Value |
| --- | --- | ---: |
| 1 | Computer | 3.447279 |
| 2 | Coffee | 2.470540 |
| 3 | Home | 1.467121 |
| 4 | Chat | 0.779510 |

### gamma = 0.50

| Rank | State | Value |
| --- | --- | ---: |
| 1 | Computer | 4.612833 |
| 2 | Coffee | 3.345404 |
| 3 | Home | 2.382290 |
| 4 | Chat | 1.766651 |

### gamma = 0.75

| Rank | State | Value |
| --- | --- | ---: |
| 1 | Computer | 7.472372 |
| 2 | Coffee | 5.917870 |
| 3 | Home | 4.891187 |
| 4 | Chat | 4.454037 |

### gamma = 1.00

| Rank | State | Value |
| --- | --- | ---: |
| 1 | Computer | 17.055272 |
| 2 | Coffee | 15.263660 |
| 3 | Home | 13.980911 |
| 4 | Chat | 13.913025 |

## Exact Infinite Horizon (gamma < 1)

### gamma = 0.00

| Rank | State | Value |
| --- | --- | ---: |
| 1 | Computer | 2.800000 |
| 2 | Coffee | 2.100000 |
| 3 | Home | 1.000000 |
| 4 | Chat | 0.300000 |

### gamma = 0.25

| Rank | State | Value |
| --- | --- | ---: |
| 1 | Computer | 3.447281 |
| 2 | Coffee | 2.470542 |
| 3 | Home | 1.467123 |
| 4 | Chat | 0.779512 |

### gamma = 0.50

| Rank | State | Value |
| --- | --- | ---: |
| 1 | Computer | 4.615788 |
| 2 | Coffee | 3.348359 |
| 3 | Home | 2.385245 |
| 4 | Chat | 1.769605 |

### gamma = 0.75

| Rank | State | Value |
| --- | --- | ---: |
| 1 | Computer | 7.813124 |
| 2 | Coffee | 6.258634 |
| 3 | Home | 5.231982 |
| 4 | Chat | 4.794797 |

## Monte Carlo (20 trajectories)

### gamma = 0.00

| Rank | State | Value [95% CI] |
| --- | --- | ---: |
| 1 | Computer | 3.000000 [1.844816, 4.155184] |
| 2 | Coffee | 2.200000 [1.930207, 2.469793] |
| 3 | Home | 1.000000 [1.000000, 1.000000] |
| 4 | Chat | 0.800000 [0.238381, 1.361619] |

### gamma = 0.25

| Rank | State | Value [95% CI] |
| --- | --- | ---: |
| 1 | Computer | 3.045312 [2.089387, 4.001236] |
| 2 | Coffee | 2.474905 [2.087097, 2.862713] |
| 3 | Home | 1.399643 [1.358559, 1.440727] |
| 4 | Chat | 1.003439 [0.194044, 1.812834] |

### gamma = 0.50

| Rank | State | Value [95% CI] |
| --- | --- | ---: |
| 1 | Computer | 4.756348 [3.218319, 6.294377] |
| 2 | Coffee | 3.584473 [2.789903, 4.379043] |
| 3 | Chat | 2.452539 [1.359560, 3.545518] |
| 4 | Home | 2.245117 [1.990176, 2.500058] |

### gamma = 0.75

| Rank | State | Value [95% CI] |
| --- | --- | ---: |
| 1 | Coffee | 5.637172 [4.368329, 6.906015] |
| 2 | Chat | 4.964242 [3.334578, 6.593905] |
| 3 | Home | 4.863923 [3.985467, 5.742379] |
| 4 | Computer | 4.762811 [2.702972, 6.822649] |

### gamma = 1.00

| Rank | State | Value [95% CI] |
| --- | --- | ---: |
| 1 | Computer | 16.250000 [13.504251, 18.995749] |
| 2 | Coffee | 13.750000 [11.075121, 16.424879] |
| 3 | Chat | 13.050000 [9.110770, 16.989230] |
| 4 | Home | 11.700000 [8.167704, 15.232296] |
