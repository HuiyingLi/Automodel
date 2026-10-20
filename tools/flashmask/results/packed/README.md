# Packed THD Qwen3-30B-A3B attention module

Forward+backward ms, lower is better. CUDA Graph, BF16, global 4K/8K, one node.
Four or sixteen unequal documents, reset RoPE positions, isolated causal attention.
See [protocol, adapters and limitations](../CP_AND_MASKS.md). Dashes mean no valid result for that cell.

## H100: 4 documents

| CP | Global seq | FlashMask Ulysses | TE ring | TE all-gather | TE all-to-all | TE 2×2 hybrid | FA3 varlen | FA4 varlen | Magi |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 4096 | 1.580 | 1.825 | — | — | — | 1.702 | 1.730 | 1.789 |
| 1 | 8192 | 3.594 | 4.182 | — | — | — | 3.854 | 3.911 | 4.095 |
| 2 | 4096 | 1.526 | 1.602 | 1.649 | 1.699 | — | — | — | 1.359 |
| 2 | 8192 | 2.925 | 2.702 | 2.772 | 2.917 | — | — | — | 2.401 |
| 4 | 4096 | 0.981 | 1.394 | 1.162 | 1.026 | 1.259 | — | — | 0.846 |
| 4 | 8192 | 1.760 | 2.397 | 2.005 | 1.972 | 2.217 | — | — | 1.678 |

Native FlashMask CP comparators (same document semantics and default NCCL environment):

| CP | Global seq | Ulysses | All-gather | CPB + all-gather | Magi |
|---|---|---|---|---|---|
| 2 | 4096 | 1.526 | 1.283 | 1.225 | 1.359 |
| 2 | 8192 | 2.925 | 2.482 | 2.419 | 2.401 |
| 4 | 4096 | 0.981 | 1.006 | — | 0.846 |
| 4 | 8192 | 1.760 | 1.811 | 1.765 | 1.678 |

## H100: 16 documents

| CP | Global seq | FlashMask Ulysses | TE ring | TE all-gather | TE all-to-all | TE 2×2 hybrid | FA3 varlen | FA4 varlen | Magi |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 4096 | 1.362 | 1.613 | — | — | — | 1.526 | 1.526 | 1.586 |
| 1 | 8192 | 2.786 | 3.179 | — | — | — | 3.076 | 3.101 | 3.229 |
| 2 | 4096 | 1.408 | 1.524 | 1.563 | 1.565 | — | — | — | 1.172 |
| 2 | 8192 | 2.531 | 2.329 | 2.332 | 2.443 | — | — | — | 1.813 |
| 4 | 4096 | 0.899 | 1.424 | 1.138 | 0.943 | 1.224 | — | — | 0.683 |
| 4 | 8192 | 1.516 | 2.223 | 1.831 | 1.688 | 2.025 | — | — | 1.274 |

Native FlashMask CP comparators (same document semantics and default NCCL environment):

| CP | Global seq | Ulysses | All-gather | CPB + all-gather | Magi |
|---|---|---|---|---|---|
| 2 | 4096 | 1.408 | 1.168 | 1.074 | 1.172 |
| 2 | 8192 | 2.531 | 2.004 | 1.870 | 1.813 |
| 4 | 4096 | 0.899 | 0.940 | — | 0.683 |
| 4 | 8192 | 1.516 | 1.540 | 1.325 | 1.274 |

## GB200: 4 documents

| CP | Global seq | FlashMask Ulysses | TE ring | TE all-gather | TE all-to-all | TE 2×2 hybrid | FA3 varlen | FA4 varlen | Magi |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 4096 | 0.987 | 1.199 | — | — | — | — | 1.080 | 0.895 |
| 1 | 8192 | 1.980 | 2.479 | — | — | — | — | 2.222 | 1.838 |
| 2 | 4096 | 1.129 | 1.172 | 1.161 | 1.234 | — | — | — | 0.737 |
| 2 | 8192 | 2.049 | 1.804 | 1.794 | 2.013 | — | — | — | 1.354 |
| 4 | 4096 | 0.763 | 1.101 | 0.860 | 0.784 | 1.027 | — | — | 0.519 |
| 4 | 8192 | 1.249 | 1.706 | 1.492 | 1.381 | 1.648 | — | — | 0.897 |

Native FlashMask CP comparators (same document semantics and default NCCL environment):

| CP | Global seq | Ulysses | All-gather | CPB + all-gather | Magi |
|---|---|---|---|---|---|
| 2 | 4096 | 1.129 | 0.915 | 0.866 | 0.737 |
| 2 | 8192 | 2.049 | 1.612 | 1.511 | 1.354 |
| 4 | 4096 | 0.763 | 0.767 | — | 0.519 |
| 4 | 8192 | 1.249 | 1.216 | 1.228 | 0.897 |

## GB200: 16 documents

| CP | Global seq | FlashMask Ulysses | TE ring | TE all-gather | TE all-to-all | TE 2×2 hybrid | FA3 varlen | FA4 varlen | Magi |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 4096 | 0.873 | 1.071 | — | — | — | — | 1.028 | 0.814 |
| 1 | 8192 | 1.640 | 2.035 | — | — | — | — | 1.947 | 1.514 |
| 2 | 4096 | 1.046 | 1.138 | 1.119 | 1.152 | — | — | — | 0.631 |
| 2 | 8192 | 1.810 | 1.599 | 1.560 | 1.755 | — | — | — | 1.067 |
| 4 | 4096 | 0.711 | 1.131 | 0.840 | 0.739 | 1.010 | — | — | 0.426 |
| 4 | 8192 | 1.102 | 1.631 | 1.384 | 1.217 | 1.529 | — | — | 0.687 |

Native FlashMask CP comparators (same document semantics and default NCCL environment):

| CP | Global seq | Ulysses | All-gather | CPB + all-gather | Magi |
|---|---|---|---|---|---|
| 2 | 4096 | 1.046 | 0.842 | 0.775 | 0.631 |
| 2 | 8192 | 1.810 | 1.340 | 1.242 | 1.067 |
| 4 | 4096 | 0.711 | 0.715 | — | 0.426 |
| 4 | 8192 | 1.102 | 1.042 | 0.992 | 0.687 |

## Coverage

Complete processes: 78. Timing rows: 608.
Worst relative RMS: 0.006558; worst normalized maximum: 0.013636.
Input hashes match across all compared backends and CP degrees for each hardware/profile/length.
