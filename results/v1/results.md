## Exact-match accuracy on held-out words

| task | wlt_only | clt_only | mtt |
|---|---|---|---|
| count | 0.650 (0.59-0.70) | 0.660 (0.60-0.71) | 0.717 (0.66-0.76) |
| spell | 0.027 (0.01-0.05) | 0.833 (0.79-0.87) | 1.000 (0.99-1.00) |
| reverse | 0.023 (0.01-0.05) | 0.773 (0.72-0.82) | 0.997 (0.98-1.00) |
| nth | 0.240 (0.20-0.29) | 0.387 (0.33-0.44) | 0.933 (0.90-0.96) |
| category | 0.980 (0.96-0.99) | 0.637 (0.58-0.69) | 1.000 (0.99-1.00) |
| **mean (letter tasks)** | 0.235 (0.21-0.26) | 0.663 (0.64-0.69) | 0.912 (0.89-0.93) |
| **mean (all tasks)** | 0.384 (0.36-0.41) | 0.658 (0.63-0.68) | 0.929 (0.92-0.94) |

Cells: accuracy (95% Wilson interval), n = 300 test prompts per task, one training seed.

## Mean sequence length in tokens (`<bos> prompt <sep> answer <eos>`)

| task | wlt_only | clt_only | mtt |
|---|---|---|---|
| count | 15.3 | 46.4 (3.03x) | 24.1 (1.58x) |
| spell | 16.4 | 43.6 (2.65x) | 29.5 (1.79x) |
| reverse | 13.1 | 39.2 (2.99x) | 23.7 (1.81x) |
| nth | 15.2 | 47.0 (3.09x) | 23.1 (1.52x) |
| category | 15.9 | 58.8 (3.70x) | 15.9 (1.00x) |
