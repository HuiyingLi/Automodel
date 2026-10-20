# FlashMask attention experiment (#4397)

[Initial dense-causal results](results/README.md) and
[expanded CP, flexible-mask and packed-THD comparisons](results/CP_AND_MASKS.md).

This branch adds the optional `backend.attn: flashmask` backend and binds its
context-parallel process group for native Qwen3-MoE attention. It uses the
PyTorch frontend of [FlashMask](https://github.com/PaddlePaddle/FlashMask), pinned
for this experiment at `5b1d0dcafd2d12e1b7df665223530fcbe140c270`.

## Supported contract

Inputs use BSHD layout and BF16/FP16 on Hopper or Blackwell. Dense causal
attention needs no mask. Callers can supply the upstream column representation
through `startend_row_indices`; this is an int32 `[batch, 1, sequence, bounds]`
tensor. Padding masks, THD/varlen metadata and window arguments are rejected
rather than silently ignored. Document masks can be expressed with column
bounds, but the packed dataset/recipe path is not integrated in this change.

For Qwen3-MoE, CP uses differentiable **Ulysses all-to-all**, with the existing
AutoModel head-tail token partition. CP must divide both Q and KV head counts
(Qwen3-30B-A3B has 32 Q heads and 4 KV heads, allowing CP 1/2/4). All-to-all
forward and backward, layout conversion and the FlashMask kernel are inside
the timed attention call. This is not FlashMask's SM100 NCCL-GIN overlap path.
TP, production packed recipes and other model families have not been validated.
The packed module benchmark below uses an explicit candidate adapter.

```yaml
model:
  backend:
    attn: flashmask
```

The existing Qwen3-MoE construction and distributed setup bind the process group.
For a standalone `Qwen3MoeAttention` module, call its existing-style
`setup_cp_attention(cp_mesh)` hook before passing local CP inputs.

## Dependency setup

Use a separate environment for FlashMask when the other backends require a
newer CuTe DSL. Start with the project's CUDA-enabled PyTorch environment:

```bash
uv pip install 'nvidia-cutlass-dsl[cu13]==4.5.2' 'nvidia-cutlass==4.2.0.0'
# The 4.5.2 base and cu13 wheels overlap: install CUDA 13's files last.
uv pip install --reinstall-package nvidia-cutlass-dsl-libs-cu13 --no-deps nvidia-cutlass-dsl-libs-cu13==4.5.2
git clone https://github.com/PaddlePaddle/FlashMask.git
cd FlashMask
git checkout 5b1d0dcafd2d12e1b7df665223530fcbe140c270
FLASHMASK_FRAMEWORK=torch FLASHMASK_BUILD=core uv pip install --no-deps --no-build-isolation -e .
```

The experiment preserves the container's torch, TE and CUDA versions. CuTe
4.6.2 is used by the container's standalone FA4 path and 4.5.2 by FlashMask.

## Reproducing the module benchmark

The repository's training benchmark and FlashMask's upstream kernel benchmark
do not compare native Qwen attention modules across these backends. The entry
point here supplies that comparison without loading a checkpoint or running a
full 30B model. It measures both the attention core and the complete native
Qwen attention module, including projections, per-head norm and RoPE.

```bash
# Select GPUs with CUDA_VISIBLE_DEVICES and use the same isolated allocation.
python -m torch.distributed.run --standalone --nproc-per-node=2   tools/flashmask/benchmark.py --backend flashmask --seq 4096 8192   --execution eager graph --output results/flashmask_cp2.json
```

Run CP 1/2/4. Backends are `flashmask`, `te`, `magi`, `fa4`, `fa3` (Hopper),
and `sdpa`. FA3/FA4 are CP=1 comparators; native PyTorch FlashAttention ring CP
is measured under `sdpa`. FA3 uses a benchmark-only callable adapter in the
otherwise unchanged Qwen module. TE is run with `NVTE_FLASH_ATTN=0` and
`NVTE_FUSED_ATTN=1` so the label identifies cuDNN fused attention. On Blackwell,
Magi requires `MAGI_ATTENTION_KERNEL_BACKEND=fa4`; Hopper uses `ffa`.
The container's Magi revision is `d7ea8afd44c790b65fab68a04a6a0fdd5adbf182`.
Its FA4 backend requires its own pinned submodule, not standalone FA4:

```bash
# In a separate environment inheriting the same container torch/TE packages:
uv pip install 'nvidia-cutlass-dsl[cu13]==4.4.2' 'apache-tvm-ffi==0.1.8.post0' 'quack-kernels==0.4.1'
uv pip install --reinstall-package nvidia-cutlass-dsl-libs-cu13 --no-deps nvidia-cutlass-dsl-libs-cu13==4.4.2
git clone https://github.com/demonatic/flash-attention.git magi-fa4
cd magi-fa4
git checkout ee1d15159cda6f3f97bfab9e487da146a8254970
uv pip install --no-deps -e flash_attn/cute
uv pip install --no-deps --no-build-isolation -e csrc/utils/magi_to_hstu -e csrc/utils/create_block_mask
```

The Qwen attention dimensions were checked against checkpoint config revision
`ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`. No checkpoint weights are loaded.

An additional Blackwell-only comparator, `flashmask_overlap`, imports the
pinned upstream `tests/torch/distributed/overlap_flashmask.py` wrapper via
`--flashmask-source /path/to/FlashMask`. Build upstream with
`FLASHMASK_BUILD=core+ovl` first. It uses the upstream NCCL-GIN CP algorithm,
with global causal masking encoded as column bounds. This is a benchmark-only
adapter, separate from the integrated Ulysses backend. Apply
`upstream-torch-overlap.patch` from this directory to the pinned FlashMask
checkout: it replaces an obsolete Paddle-importing compatibility path with the
same framework-neutral fwd/bwd functions. No kernel or communication logic is
changed. The source diff hash is recorded. Apply this zero-context patch with:

```bash
git -C /path/to/FlashMask apply --unidiff-zero /path/to/Automodel/tools/flashmask/upstream-torch-overlap.patch
```

Run each sequence and scope in a fresh process as required by upstream buffer
lifetimes. Its compiled overlap implementation only accepts local lengths
4096/8192/16384/32768/65536/131072. Therefore among the requested global 4K/8K
and CP 2/4 settings, only global 8K at CP 2 is supported. Mask rebasing/reordering
inside the upstream wrapper is included in timing. This restriction does not
apply to the integrated Ulysses backend.

Every invocation first compares outputs, input gradients and module parameter
gradients with an identical full-sequence SDPA module. Parameter gradients are
summed across CP for this correctness check, outside timing. Seeded inputs,
weights, random upstream gradients and global token order are shared across
CP degrees; Magi uses its own native dispatch/key API. Input dispatch, reference
execution, JIT compilation and parameter-gradient synchronization are outside
timing. Sequence length is the global token count, not per-rank length. Without a mask
profile it is one causal sequence; document profiles preserve separate boundaries.

JSON records per-rank parity, input hash, actual dependency versions, hardware,
source revision/diff, timing samples and trial summaries. Timings reduce the
maximum across CP ranks. Eager CUDA events and wall time include normal Python
launch behavior. Optional CUDA Graph rows capture five steps per replay to
expose GPU execution cost and verify replay output/input gradients. Unsupported
capture is an explicit failure, not an eager timing mislabeled as graph.

The `sdpa` ring comparator uses PyTorch's public `context_parallel` manager;
its buffer API requires a sequence buffer even when inputs are already
partitioned, so a separate coordinate buffer is sharded/restored by that
manager. Eager timings include that manager's bookkeeping.

## CP algorithms, flexible masks and packed THD

TE defaults to `--te-cp-comm p2p` (KV ring). Other choices are `all_gather`,
`a2a` (Ulysses), and `a2a+p2p`. The hybrid comparator requires four ranks
and explicitly forms 2×2 groups; it is a single-node test.

`--mask documents` and `--mask prefix` select four unequal documents
`[S/8, S/8, S/4, S/2]`. Prefix-LM makes each document's first half
bidirectional. In the ordinary BSHD interface, FA3/TE use varlen decomposition
at CP1; FA4 supports either `--fa4-mask-engine sparse` (including backward
block-sparse metadata) or `varlen`. `--fa4-unpacked` disables GQA packing
in the sparse comparator. Neither mode materializes a dense attention mask
inside the timed call.

`--packed --documents 4` or `16` selects actual THD input to the native Qwen
module, resets positions per document and passes cumulative sequence lengths.
The sixteen-document lengths repeat `[S/32, S/32, S/16, S/8]` four times.
Use `--scope module --mask documents`. These aligned synthetic profiles have
no physical padding; they do not establish arbitrary ragged/padded recipe support.

~~~bash
# TE ring with true THD packing. Swap the backend for flashmask or magi.
NVTE_FLASH_ATTN=0 NVTE_FUSED_ATTN=1 python -m torch.distributed.run --standalone --nproc-per-node=2 tools/flashmask/benchmark.py --backend te --te-cp-comm p2p --packed --documents 16 --mask documents --scope module --seq 4096 8192 --execution graph --output results/packed_te_cp2.json

# TE 2×2 hybrid; this is attention CP, not the MoE HybridEP dispatcher.
NVTE_FLASH_ATTN=0 NVTE_FUSED_ATTN=1 python -m torch.distributed.run --standalone --nproc-per-node=4 tools/flashmask/benchmark.py --backend te --te-cp-comm a2a+p2p --packed --documents 4 --mask documents --scope module --seq 4096 8192 --execution graph --output results/packed_te_hybrid.json
~~~

On GB200 with the pinned FlashMask/CuTe 4.5.2 stack, the THD adapter exposed
SM100 backward JIT errors from variables first assigned in dynamic branches.
Apply [upstream-sm100-mask-init.patch](upstream-sm100-mask-init.patch) to that
FlashMask checkout before reproducing these packed results. It moves loop
bounds, constant mask flags and a static TMA copy descriptor before their
dynamic guards; attention/masking and communication semantics are unchanged.
This additional patch is distinct from the import-only CP wrapper patch.

~~~bash
git -C /path/to/FlashMask apply --unidiff-zero /path/to/Automodel/tools/flashmask/upstream-sm100-mask-init.patch
~~~

The packed FlashMask adapter in [packed.py](packed.py) adds a batch view around
THD tensors and a differentiable sequence/head exchange using TE's per-document
head-tail token partition. Its reordering and communication are timed.
TE/Magi use native THD calls; FA3/FA4 use native varlen kernels. The adapter
changes the benchmark's attention interface only; it keeps Qwen projections,
normalization, RoPE and output projection. It does **not** enable packed SFT
in the production backend.

`flashmask_allgather` and `flashmask_cpb` import the pinned upstream Torch
comparators through `--flashmask-source`. Apply the same import-only patch
used by the overlap comparator. CPB additionally requires its native extension;
build with upstream `FLASHMASK_BUILD=core+cpb` or its `flashmask/cp_balance`
CMake target. The experiment uses the default 2048-token chunks: total sequence
length must divide by `2048 * CP`, so 4K/CP4 has no CPB result.

~~~bash
NVTE_FLASH_ATTN=0 NVTE_FUSED_ATTN=1 python -m torch.distributed.run --standalone --nproc-per-node=2 tools/flashmask/benchmark.py --backend flashmask_cpb --flashmask-source /path/to/FlashMask --packed --documents 16 --mask documents --scope module --seq 4096 8192 --execution graph --output results/packed_flashmask_cpb.json

python tools/flashmask/validate_masks.py
~~~

## Validation

```bash
python -m pytest tests/unit_tests/attention/test_flashmask_attention.py -q
python -m pytest tests/functional_tests/attention/test_flashmask_attention.py -q
```

CPU tests use an independent FP32 softmax oracle with real two-process Gloo
all-to-all, check Q/K/V gradients and include a deliberately wrong token-order
negative control. CUDA tests compare actual kernels against FP32 matrix-multiply
and softmax for causal, document-boundary and fully masked cases. The benchmark
adds real NCCL CP checks at the requested global lengths. These are component
checks, not checkpoint-loading, full-model or end-to-end training validation.
