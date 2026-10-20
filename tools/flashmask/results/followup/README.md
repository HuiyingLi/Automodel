# CP algorithms and flexible masks: Qwen3-30B-A3B attention module

Module forward+backward latency in ms; lower is better. CUDA Graph unless a table explicitly says eager. Global sequence lengths; BF16, batch 1, 32 Q / 4 KV heads, head dimension 128. See [protocol and interpretation](../CP_AND_MASKS.md).

## H100: causal CP algorithms

| CP | Global seq | FlashMask Ulysses | TE ring | TE all-gather | TE all-to-all | TE 2×2 hybrid | Magi |
|---|---|---|---|---|---|---|---|
| 2 | 4096 | 1.716 | 1.616 | 1.789 | 1.915 | — | 1.825 |
| 2 | 8192 | 3.812 | 3.549 | 3.933 | 3.973 | — | 3.867 |
| 4 | 4096 | 1.090 | 1.173 | 1.167 | 1.137 | 1.207 | 1.191 |
| 4 | 8192 | 2.160 | 2.363 | 2.411 | 2.442 | 2.512 | 2.561 |

## H100: documents, CP1

| Global seq | FlashMask | TE varlen | FA3 varlen | FA4 sparse, GQA packed | FA4 sparse, unpacked | FA4 varlen | Magi |
|---|---|---|---|---|---|---|---|
| 4096 | 1.580 | 1.989 | 1.711 | 1.765 | 1.765 | 1.724 | 1.790 |
| 8192 | 3.600 | 4.947 | 3.873 | 4.040 | 4.050 | 3.911 | 4.091 |

## H100: documents, CP2/4

| CP | Global seq | FlashMask Ulysses | Magi |
|---|---|---|---|
| 2 | 4096 | 1.545 | 1.363 |
| 2 | 8192 | 2.937 | 2.392 |
| 4 | 4096 | 1.014 | 0.847 |
| 4 | 8192 | 1.778 | 1.679 |

## H100: prefix, CP1

| Global seq | FlashMask | TE varlen | FA3 varlen | FA4 sparse, GQA packed | FA4 sparse, unpacked | FA4 varlen | Magi |
|---|---|---|---|---|---|---|---|
| 4096 | 1.628 | 2.461 | 2.263 | 1.808 | 1.806 | 2.254 | 1.863 |
| 8192 | 3.824 | 5.524 | 5.044 | 4.240 | 4.224 | 5.024 | 4.352 |

## H100: prefix, CP2/4

| CP | Global seq | FlashMask Ulysses | Magi |
|---|---|---|---|
| 2 | 4096 | 1.563 | 1.445 |
| 2 | 8192 | 3.028 | 2.592 |
| 4 | 4096 | 1.019 | 0.912 |
| 4 | 8192 | 1.816 | 1.832 |

## H100: upstream FlashMask CP, global 8K

| Mask | CP | Ulysses graph | All-gather graph | CPB + all-gather graph | Ulysses eager | All-gather eager | CPB eager | GIN overlap eager | Magi eager |
|---|---|---|---|---|---|---|---|---|---|
| causal | 2 | 3.812 | 3.384 | 3.333 | 3.830 | 3.765 | 3.463 | — | 4.055 |
| causal | 4 | 2.160 | 2.171 | 2.489 | 3.470 | 4.334 | 4.056 | — | 3.409 |
| prefix | 2 | 3.028 | 2.504 | 2.446 | 5.259 | 3.748 | 3.484 | — | 2.898 |
| prefix | 4 | 1.816 | 1.814 | 1.764 | 4.233 | 3.988 | 3.561 | — | 2.965 |
## GB200: causal CP algorithms

| CP | Global seq | FlashMask Ulysses | TE ring | TE all-gather | TE all-to-all | TE 2×2 hybrid | Magi |
|---|---|---|---|---|---|---|---|
| 2 | 4096 | 1.188 | 1.019 | 1.113 | 1.261 | — | 0.888 |
| 2 | 8192 | 2.304 | 1.891 | 1.956 | 2.216 | — | 1.840 |
| 4 | 4096 | 0.827 | 0.863 | 0.764 | 0.770 | 0.867 | 0.605 |
| 4 | 8192 | 1.404 | 1.535 | 1.540 | 1.445 | 1.630 | 1.177 |

## GB200: documents, CP1

| Global seq | FlashMask | TE varlen | FA3 varlen | FA4 sparse, GQA packed | FA4 sparse, unpacked | FA4 varlen | Magi |
|---|---|---|---|---|---|---|---|
| 4096 | 0.987 | 1.195 | — | 1.253 | 1.297 | 1.083 | 0.898 |
| 8192 | 1.987 | 2.457 | — | 2.542 | 2.630 | 2.227 | 1.849 |

## GB200: documents, CP2/4

| CP | Global seq | FlashMask Ulysses | Magi |
|---|---|---|---|
| 2 | 4096 | 1.126 | 0.739 |
| 2 | 8192 | 1.990 | 1.356 |
| 4 | 4096 | 0.788 | 0.518 |
| 4 | 8192 | 1.251 | 0.896 |

## GB200: prefix, CP1

| Global seq | FlashMask | TE varlen | FA3 varlen | FA4 sparse, GQA packed | FA4 sparse, unpacked | FA4 varlen | Magi |
|---|---|---|---|---|---|---|---|
| 4096 | 0.992 | 1.614 | — | 1.253 | 1.286 | 1.507 | 0.906 |
| 8192 | 2.051 | 3.394 | — | 2.580 | 2.629 | 3.137 | 1.944 |

## GB200: prefix, CP2/4

| CP | Global seq | FlashMask Ulysses | Magi |
|---|---|---|---|
| 2 | 4096 | 1.132 | 0.752 |
| 2 | 8192 | 2.031 | 1.419 |
| 4 | 4096 | 0.792 | 0.531 |
| 4 | 8192 | 1.271 | 0.914 |

## GB200: upstream FlashMask CP, global 8K

| Mask | CP | Ulysses graph | All-gather graph | CPB + all-gather graph | Ulysses eager | All-gather eager | CPB eager | GIN overlap eager | Magi eager |
|---|---|---|---|---|---|---|---|---|---|
| causal | 2 | 2.304 | 2.009 | 1.896 | 3.937 | 4.359 | 3.789 | 3.953 | 3.566 |
| causal | 4 | 1.404 | 1.419 | 1.581 | 4.061 | 4.594 | 3.879 | — | 3.639 |
| prefix | 2 | 2.031 | 1.622 | 1.522 | 4.781 | 4.349 | 3.813 | 3.932 | 3.527 |
| prefix | 4 | 1.271 | 1.211 | 1.228 | 4.911 | 4.593 | 3.902 | — | 3.651 |
