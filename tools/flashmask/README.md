# FlashMask attention experiment (#4397)

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
TP, packed CP and other model families have not been validated.

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
4.6.2 is used by the container's FA4/Magi paths and 4.5.2 by FlashMask.

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

Every invocation first compares outputs, input gradients and module parameter
gradients with an identical full-sequence SDPA module. Parameter gradients are
summed across CP for this correctness check, outside timing. Seeded inputs,
weights, random upstream gradients and global token order are shared across
CP degrees; Magi uses its own native dispatch/key API. Input dispatch, reference
execution, JIT compilation and parameter-gradient synchronization are outside
timing. Sequence length means one global causal sequence, not per-rank length.

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
