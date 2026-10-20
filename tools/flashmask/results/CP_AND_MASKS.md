# FlashMask: CP algorithm and mask follow-up

The initial integration chose Ulysses to reuse AutoModel's head-tail input
partitioning through one differentiable PyTorch path on H100 and GB200. That
was an implementation choice, not a performance-selected CP algorithm. Qwen's
32 query heads and only 4 KV heads make this distinction material: Ulysses
exchanges Q/K/V and the attention output; a KV ring sends the much smaller
KV tensors. Actual latency also depends on message size, scheduling, kernel
shape and overlap, so neither algorithm wins universally.

The previous report's TE `p2p` baseline **was TE ring attention**. This follow-up
labels it explicitly and adds TE `all_gather`, `a2a` (Ulysses), and a 2×2
`a2a+p2p` hierarchy. All tests use one node; the hierarchy does not test an
NVLink-within-node / IB-between-nodes topology. Algorithm definitions were
checked in the executed TE 2.19.0 source and the
[TE API documentation](https://docs.nvidia.com/deeplearning/transformer-engine/api/pytorch.html).

[Complete measured tables](followup/README.md), [timings CSV](followup/timings.csv),
and [source/environment manifest](followup/manifest.json) preserve the expanded
comparison. They supplement the original dense-causal results rather than
changing that experiment's historical data.

## Observed results

For dense causal CP, TE ring is faster than the integrated FlashMask Ulysses
path at CP2 on both GPUs. At CP4, FlashMask beats all four TE communication
choices on H100; GB200 4K favors TE all-gather/all-to-all over FlashMask, while
Magi is faster than both. The single-node TE 2×2 hierarchy does not improve
over its best flat communication mode in this matrix.

Native FlashMask CP changes the comparison. At 8K/CP2 on H100, CPB + all-gather
takes 3.333 ms versus Ulysses 3.812, TE ring 3.549 and Magi 3.867 ms. On GB200,
the corresponding values are 1.896, 2.304, 1.891 and 1.840 ms: CPB is effectively
tied with TE ring, and Magi remains faster. CPB and plain all-gather are also
close in several cells, so this does not establish a general benefit from
workload balancing alone. The native-driver NCCL environment differs as
disclosed below.

For four-document masks without THD packing, FlashMask wins CP1 on H100,
including against FA3 varlen and both FA4 sparse GQA settings. On GB200 it
beats TE/standalone FA4 for document and prefix masks, but Magi is faster.
For prefix masks at 8K/CP1, FlashMask takes 3.824 ms on H100 versus the best
FA4 setting's 4.224 ms and TE's 5.524 ms; on GB200 it takes 2.051 ms versus
FA4's 2.580 ms, TE's 3.394 ms and Magi's 1.944 ms. This demonstrates a useful
flexible-mask case without implying FlashMask always wins with complex masks.

The BSHD flexible-mask comparisons contain 82 complete process runs and
580 timing rows (forward/forward+backward, eager/graph). All passed the
unchanged output, input-gradient and summed parameter-gradient gates; the
largest relative RMS error is 0.00630. Setup failures and interrupted runs
are excluded from these counts.

## What FlashMask offers, and what this experiment exercises

The [pinned upstream source](https://github.com/PaddlePaddle/FlashMask/tree/5b1d0dcafd2d12e1b7df665223530fcbe140c270)
uses column boundaries to describe supported masks, skips fully masked blocks,
and applies element masking on partial blocks. Its distributed work also adds
mask-based workload assignment, KV pruning and communication overlap.
These mechanisms are separate: using its kernel behind Ulysses does not invoke
its CPB scheduler or its GIN overlap implementation.

This comparison exercises document isolation and document-local Prefix-LM in
addition to ordinary causal attention. Four document lengths are
`[S/8, S/8, S/4, S/2]`; each Prefix-LM prefix is half its document. Prefix queries
attend the whole prefix; suffix queries attend the prefix and previous suffix
tokens. Cross-document attention is forbidden. Allowed attention-pair density
is approximately 17.2% for document causal and 21.5% for Prefix-LM, relative to a
full square. These are synthetic module inputs, not a packed training recipe.

A two-boundary int32 mask takes 32 KiB at 4K or 64 KiB at 8K (batch 1, shared
across heads). A square boolean mask would take 16/64 MiB. This is a
representation-size calculation, **not** a claim that TE, FA4 or Magi normally
allocate dense masks: varlen metadata, block-sparse metadata and Magi ranges
are also compact. The tested Qwen head dimension is 128; it does not exercise
FlashMask's support for larger head dimensions. Nor does 4K/8K establish the
upstream long-context scaling claims.

## Backends and fair mask comparisons

- FlashMask uses the same column mask and native kernel, with the integrated
  Ulysses path at CP2/4. The production integration remains this path.
- Magi uses native flexible attention ranges and its native input dispatcher
  and CP communication. Its range key is passed explicitly to `calc_attn`.
- FA4 tests `mask_mod` **with forward and backward block sparsity**, including
  both GQA packing choices. It also tests its native varlen API.
- TE and FA3 use their native varlen APIs. Ordinary document causal needs one
  varlen call. Prefix-LM is decomposed into two disjoint query sets: noncausal
  prefix Q against prefix KV, and bottom-right causal suffix Q against full
  document KV. No attention pairs are computed twice. The Q/K/V packing and
  output restoration for that decomposition are included in module timing.
- The TE/FA varlen-decomposition adapter covers CP1. Missing flexible-mask CP
  cells are not evidence that TE can never support those masks with CP.
- Standalone FA3 is measured on H100. Magi's GB200 FA4 fork and standalone FA4
  are different pinned implementations; package/source details match the
  [original dependency report](README.md#dependencies-and-correctness).

The benchmark adapts interfaces around the actual `Qwen3MoeAttention`; it does
not replace its projections, norms, RoPE or output projection. Mask construction,
FA4 sparse-metadata preparation, Magi dispatch planning/input distribution,
and CPB planning/input redistribution are outside timing. They are preparation
costs and may need amortization in training with changing masks. Compilation,
reference execution and parameter-gradient synchronization are also excluded.
Attention and CP communication are included. The upstream dual-chunk and GIN
wrappers also rebase/reorder their mask inside the timed call; CPB planning
and its mask redistribution are prepared outside the timed call. These conventions apply to every
backend; eager and CUDA Graph results are kept separate.

## Upstream FlashMask CP comparators

`flashmask_allgather` runs the upstream dual-chunk all-gather reference;
`flashmask_cpb` runs upstream LPT workload assignment followed by its balanced
all-gather wrapper. Both gather full KV and reduce-scatter its gradients; CPB
alone does not prune KV communication. Their only upstream Python patch selects
the framework-neutral kernel imports instead of importing the Paddle frontend.
The patch is saved in [upstream-torch-overlap.patch](../upstream-torch-overlap.patch).

CPB uses its upstream default 2048-token balance chunks. At global 8K/CP4
there is only one chunk per rank, so there is no freedom to equalize work by
combining chunks. That short-sequence case must not be extrapolated to finer
chunks or much longer contexts. The exposed nondefault balance-chunk parameter
was not used: source inspection found the reranking call does not forward it.

The GIN overlap comparator is the upstream Torch wrapper on GB200. It requires
local sequence lengths from a fixed supported set beginning at 4096, so among
the original 4K/8K CP2/4 matrix it can run only global 8K/CP2. Its current graph
capture fails; only eager results are compared. The upstream CP driver uses
`NCCL_NVLS_ENABLE=0` and the upstream overlap environment settings; this differs
from the default NCCL environment of the integrated-path matrix and is recorded
with the raw artifacts. No SM90 GIN result is claimed from the SM100 Torch path.

## Actual packed THD module comparison

[Packed tables](packed/README.md), [CSV](packed/timings.csv), and
[manifest](packed/manifest.json) are a separate experiment from the BSHD
document-mask tables above. The Qwen module consumes `[tokens, hidden]`;
TE/Magi receive THD Q/K/V, and FA3/FA4 use native varlen APIs. FlashMask
uses the explicit candidate adapter in [packed.py](../packed.py), adding a
no-copy batch view and, for Ulysses, differentiable exchanges/permutations
that restore logical document order. TE and this adapter use TE's native
per-document head-tail partition; Magi and native FlashMask CP use their
own dispatch layouts. All compare identical logical inputs and document masks.

The packed results preserve the hardware-dependent ranking:

- **H100, CP1:** FlashMask is fastest for both document profiles and lengths.
  At 8K/four documents it takes 3.594 ms versus FA3 3.854, FA4 3.911,
  TE 4.182 and Magi 4.095 ms.
- **H100, CP2:** native CPB improves materially over Ulysses. At 4K it beats
  Magi for both profiles (four documents: 1.225 vs 1.359 ms; sixteen:
  1.074 vs 1.172 ms). At 8K/four documents it is near-tied with Magi
  (2.419 vs 2.401 ms), while Magi is faster with sixteen documents
  (1.813 vs CPB 1.870 ms). Magi is fastest at CP4.
- **GB200:** Magi is fastest in every tested packed profile/length/CP cell.
  FlashMask still beats TE and standalone FA4 at CP1. Native CPB narrows
  its CP2 gap: at 8K/four documents, Ulysses is 2.049 ms, CPB 1.511 ms,
  and Magi 1.354 ms. At sixteen documents these are 1.810, 1.242 and
  1.067 ms.
- **TE hybrid:** the single-node 2×2 hierarchy does not beat the best flat
  TE communication mode in these packed cases. This does not establish its
  ranking across nodes or across heterogeneous links.

All 78 final packed process runs (608 timing rows) pass the unchanged
correctness gates. The largest relative RMS error is 0.006558 and the largest
normalized maximum error is 0.013636. Hashes match across backends and CP
degrees for each logical input/profile/length. These numbers are full-module
forward+backward CUDA Graph latency; forward-only and eager measurements
remain in the CSV.

Two profiles keep the global token count fixed: four documents
`[S/8, S/8, S/4, S/2]`, or sixteen documents repeating
`[S/32, S/32, S/16, S/8]` four times. Positions reset to zero at each
document boundary. There is no physical padding in these aligned profiles.
No text dataset, shuffle, tokenizer or sample packing is involved: inputs
are fixed seeded module activations, and the boundary lists and input hashes
are saved. Every backend and CP degree reuses the same logical example.

The final packed tables use matching default NCCL settings across all
backends, including native FlashMask CP (`NCCL_NVLS_ENABLE` unset). The
earlier H100 native-CP run with NVLS disabled is preserved separately in
`runs/packed_native_nvls0` and excluded from these tables.

Projections, per-head normalization, RoPE, attention, output projection and
attention CP traffic are timed. The FlashMask Ulysses token permutations
are included; document metadata and initial backend dispatch are prepared
once. Eager and graph modes retain the same three-trial protocol. TE ring,
all-gather, all-to-all and its 2×2 hybrid all run with real THD metadata.

The GB200 packed runs use an additional
[SM100 initialization patch](../upstream-sm100-mask-init.patch) on the pinned
FlashMask source. Unmodified THD runs failed during backward CuTe compilation
(`loop_start`, then mask-flag/copy-descriptor scope); a BSHD control passed.
Importing FlashMask before TE did not resolve the failure. The patch lifts
loop-bound initialization, constant mask flags, and a static TMA copy
descriptor outside dynamic guards. It does not change attention pairs,
arithmetic, kernel tiling, work scheduling or communication. The same packed
smoke that failed before the patch passes output/gradient and graph-replay
checks afterward. The patched GB200 build also passes all three existing
CUDA kernel/oracle tests (causal, document and empty masks), a separate
4K/8K BSHD Prefix-LM module control, and the four/sixteen-document FP32
mathematical-oracle checks. Full packed measurements record the resulting upstream
source/diff hash. H100 uses the original SM90 kernels without this patch.

Failure logs are retained in `runs/packed/initial_failed` and
`runs/packed_debug`; diagnostics with shortened sampling are excluded from
the timing tables. The older BSHD tables above retain their original
unpatched-source provenance.

This validates the benchmark adapter's module computation and gradients.
Production FlashMask still rejects THD/varlen metadata; native packed SFT,
checkpoint loading, FSDP/EP, and full-model training remain unvalidated.
HybridCP here is TE's attention hierarchy; AutoModel's `hybridep` is a
separate MoE dispatcher and is not an attention backend.

## Protocol and validation

Hardware and all model dimensions match the original request. DFW follow-up
runs use H100 GPUs 4–7 on `pool0-00845`, interactive allocation `20562116`.
HSG initially reused GB200 allocation `8094720`; after its scheduled expiry,
allocation `8099693` (13:37–17:37 PDT) uses the same node `nvl72176-T02` and
the same four GPU UUIDs. The replacement has healthy completed fabric
initialization, CliqueId 8558 and ClusterUUID
`31402bd2-07b6-4797-a9cc-9d7bb5f2d211`. Its FA4 unpacked-GQA and packed
measurements are identified in the artifacts. Benchmarks execute sequentially on each device group.

Each process uses warmup 10, 3 trials × 30 samples, maximum latency across
ranks, and reports the median of trial medians. Graph replay contains five
steps. Both outputs and input gradients are compared to an independent
per-document SDPA reference, and module parameter gradients are summed across
CP ranks before the eager comparison. Graph replay separately checks outputs
and input gradients. The original 2% relative RMS / 3% normalized-max
gates are unchanged. The small mathematical-oracle check independently verifies
the reference and column-mask encoding against FP32 explicit attention and
checks that incorrectly using a single global causal mask fails strongly.

Raw logs and JSON live under each cluster's
`/lustre/fsw/portfolios/coreai/users/huiyingl/flashmask4397/runs/{cp_algorithms,masks,upstream_cp}`.
The manifest contains source revisions, source/file hashes when recorded,
package versions, and the native CPB assignment. Initial FA4 adapter errors
(CuTe compile-time branching and unpacking its return tuple) were corrected;
failed setup attempts are preserved and do not count as performance evidence.

Coverage is attention-module forward/backward and communication correctness.
It does not validate a loaded 30B checkpoint, MoE layers, full-model training,
native packed-data recipes, or multi-node scaling.


## Archived evidence

The full raw logs/JSON, failed diagnostics, drivers, dependency build scripts,
upstream diffs, native-library hashes, hardware manifests, and benchmark
source snapshots are archived on DFW:

`/lustre/fsw/portfolios/coreai/users/huiyingl/flashmask4397/artifacts/flashmask_cp_masks_packed_20261020.tar.gz`

SHA-256: `fa9a748b9e00a0d45bec1886ea768d54bf5cb4d2b3b06f18766092085c335deb`.

All 78 packed runs' benchmark/helper hashes were checked against the archived
`packed_source_initial` or `packed_source_validated` snapshots. Later CLI
help/error wording changes do not alter timed execution. The independent
four/sixteen-document oracle's largest relative RMS error is 0.00357; using
an incorrect global causal mask produces relative RMS above 0.81 on both
GPUs. Raw HSG evidence is mirrored under DFW's `runs/hsg_followup`; original
artifacts remain on HSG. The archive excludes the generated report directory
so that its hash can be recorded here without a circular dependency.
