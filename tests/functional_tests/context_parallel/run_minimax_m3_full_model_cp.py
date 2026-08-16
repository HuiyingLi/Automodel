#!/usr/bin/env python
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Full-model CP forward-equivalence for MiniMax M3, using the REAL recipe sharding.

Unlike run_minimax_m3_sparse_cp.py (which shards one attention layer with the same slot
function it reconstructs with -- self-consistent but blind to layout mismatches), this
drives the WHOLE model exactly as recipes/vlm/finetune.py does under CP:

  apply_cp(model, cp_mesh)                      # sets _cp_mesh on every attention layer
  cp_sharder = ContextParallelSharder(None, ...)
  train_ctx, batch = cp_sharder.shard(batch)          # torch context_parallel shards the seq
  with train_ctx(): logits_local = model(**batch)
  logits_full = context_parallel_unshard(...)   # undo the load-balanced layout

then asserts the reconstructed per-token logits match the eager full-sequence logits.
This exercises dense + sparse model-owned CP attention and RoPE positions against the
ACTUAL context_parallel layout. ``NEMO_M3_CP_TEST_PACKED=1`` additionally resets
position ids at document boundaries and compares against eager block-diagonal attention,
which catches cross-document leakage in any layer.

Usage:
    torchrun --nproc_per_node=2 --master_port=29551 \
        tests/functional_tests/context_parallel/run_minimax_m3_full_model_cp.py

    NEMO_M3_CP_TEST_PACKED=1 torchrun --nproc_per_node=2 --master_port=29551 \
        tests/functional_tests/context_parallel/run_minimax_m3_full_model_cp.py
"""

import os
import sys

import torch
import torch.distributed as dist
import torch.nn.functional as F

_FLEX_CFG = dict(
    hidden_size=256,
    intermediate_size=64,
    dense_intermediate_size=128,
    shared_intermediate_size=64,
    num_hidden_layers=3,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=64,
    rotary_dim=32,
    partial_rotary_factor=0.5,
    vocab_size=128,
    max_position_embeddings=4096,
    rms_norm_eps=1e-6,
    rope_theta=10000.0,
    num_local_experts=4,
    num_experts_per_tok=2,
    n_shared_experts=1,
    moe_layer_freq=[0, 1, 1],
    use_gemma_norm=True,
    use_qk_norm=True,
    qk_norm_type="per_head",
    scoring_func="sigmoid",
    use_routing_bias=True,
    routed_scaling_factor=2.0,
    swiglu_alpha=1.702,
    swiglu_limit=7.0,
    num_mtp_modules=0,
    sparse_attention_config=dict(
        use_sparse_attention=True,
        sparse_index_dim=64,
        sparse_num_index_heads=2,
        sparse_topk_blocks=2,
        sparse_block_size=128,
        sparse_score_type="max",
        sparse_init_block=0,
        sparse_local_block=1,
        sparse_attention_freq=[0, 1, 1],
        sparse_disable_index_value=[0, 1, 1],
    ),
)


def _logits(out):
    return out.logits if hasattr(out, "logits") else out


def main():
    if not ("RANK" in os.environ and "WORLD_SIZE" in os.environ):
        print("ERROR: launch with torchrun (needs RANK/WORLD_SIZE/LOCAL_RANK).", file=sys.stderr)
        sys.exit(1)

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    cp_size = world_size

    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor.experimental._attention import context_parallel_unshard

    from nemo_automodel.components.distributed.context_parallel import ContextParallelSharder
    from nemo_automodel.components.models.common import BackendConfig
    from nemo_automodel.components.models.minimax_m3_vl.config import MiniMaxM3VLTextConfig
    from nemo_automodel.components.models.minimax_m3_vl.model import MiniMaxM3SparseForCausalLM
    from nemo_automodel.components.moe.parallelizer import apply_cp

    backend = BackendConfig(
        linear="torch",
        attn="sdpa",
        rms_norm="torch",
        experts="torch",
        dispatcher="torch",
        rope_fusion=False,
        fake_balanced_gate=False,
        enable_hf_state_dict_adapter=False,
    )

    # bf16 matches the production recipe. fp32 is useful for tighter kernel parity.
    # Localization toggles:
    #   NEMO_M3_CP_TEST_DENSE_ONLY=1 -> all layers dense.
    #   NEMO_M3_CP_TEST_ALL_SPARSE=1 -> all layers sparse.
    #   NEMO_M3_CP_TEST_FP32=1       -> fp32.
    #   NEMO_M3_CP_TEST_PACKED=1     -> four packed docs with block-diagonal reference mask.
    cfg_kwargs = dict(_FLEX_CFG)
    dense_only = bool(os.environ.get("NEMO_M3_CP_TEST_DENSE_ONLY"))
    all_sparse = bool(os.environ.get("NEMO_M3_CP_TEST_ALL_SPARSE"))
    fp32 = bool(os.environ.get("NEMO_M3_CP_TEST_FP32"))
    packed = bool(os.environ.get("NEMO_M3_CP_TEST_PACKED"))
    sac = dict(cfg_kwargs["sparse_attention_config"])
    if dense_only:
        sac["sparse_attention_freq"] = [0, 0, 0]
        sac["use_sparse_attention"] = False
    elif all_sparse:
        sac["sparse_attention_freq"] = [1, 1, 1]
        sac["sparse_disable_index_value"] = [1, 1, 1]  # selection-only indexer on every layer
    cfg_kwargs["sparse_attention_config"] = sac
    dtype = torch.float32 if fp32 else torch.bfloat16
    if rank == 0:
        print(f"[localize] dense_only={dense_only} all_sparse={all_sparse} packed={packed} dtype={dtype}")

    torch.manual_seed(0)
    cfg = MiniMaxM3VLTextConfig(torch_dtype=("float32" if fp32 else "bfloat16"), **cfg_kwargs)
    model = MiniMaxM3SparseForCausalLM(cfg, backend=backend).eval().to(device)
    model.initialize_weights(dtype=dtype)
    model.to(dtype)
    for p in model.parameters():
        dist.broadcast(p.data, src=0)
    for b in model.buffers():
        dist.broadcast(b.data, src=0)

    bsz = 1
    seqlen = 256 * cp_size  # 2*cp_size blocks (block_size=128); divisible by 2*cp -> no pad
    input_ids = torch.randint(1, cfg.vocab_size, (bsz, seqlen), device=device)
    dist.broadcast(input_ids, src=0)
    attention_mask = None
    seq_lens = None
    if packed:
        num_docs = 4
        doc_len = seqlen // num_docs
        slots = torch.arange(seqlen, device=device)
        doc_ids = slots // doc_len
        seq_lens = torch.full((bsz, num_docs), doc_len, dtype=torch.long, device=device)
        position_ids = (slots % doc_len).unsqueeze(0)
        attention_mask = ((doc_ids[:, None] == doc_ids[None, :]) & (slots[:, None] >= slots[None, :]))[None, None]
    else:
        position_ids = torch.arange(seqlen, device=device).unsqueeze(0)

    # ---- Reference: eager full-sequence forward (CP off; sparse layers _cp_mesh=None) ----
    with torch.no_grad():
        logits_eager = _logits(
            model(input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask)
        ).float()  # [1, T, V]

    # ---- CP: exactly the recipe path ----
    device_mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("cp",))
    apply_cp(model, device_mesh["cp"])  # sets _cp_mesh on every self_attn

    batch = {
        "input_ids": input_ids.clone(),
        "labels": input_ids.clone(),  # required by the CP dispatch
        "position_ids": position_ids.clone(),
    }
    if attention_mask is not None:
        batch["attention_mask"] = attention_mask.clone()
        batch["seq_lens"] = seq_lens.clone()
    cp_sharder = ContextParallelSharder(None, device_mesh, batch)
    train_ctx, batch = cp_sharder.shard(batch)
    with torch.no_grad(), train_ctx():
        logits_local = _logits(
            model(
                input_ids=batch["input_ids"],
                position_ids=batch["position_ids"],
                attention_mask=batch.get("attention_mask"),
                padding_mask=batch.get("padding_mask"),
                seq_lens=batch.get("seq_lens"),
            )
        ).float()

    (logits_cp_full,) = context_parallel_unshard(device_mesh["cp"], [logits_local], seq_dims=[1])
    logits_cp_full = logits_cp_full[:, :seqlen, :]  # drop any cp padding (none expected here)

    if rank == 0:
        diff = (logits_cp_full - logits_eager).abs()
        mean_abs = diff.mean().item()
        max_abs = diff.max().item()
        per_pos = diff.mean(dim=(0, 2))  # [T]; leakage concentrates at early positions
        worst = torch.topk(per_pos, k=min(8, seqlen)).indices.tolist()
        # Per-token argmax agreement is the leakage-sensitive metric.
        top1 = (logits_cp_full.argmax(-1) == logits_eager.argmax(-1)).float().mean().item()
        # Causal-LM CE on identical tokens. Ignore document starts because the token
        # before a packed boundary must not predict the first token of the next doc.
        targets = input_ids[:, 1:]
        valid = position_ids[:, 1:] != 0
        loss_eager = F.cross_entropy(logits_eager[:, :-1][valid], targets[valid]).item()
        loss_cp = F.cross_entropy(logits_cp_full[:, :-1][valid], targets[valid]).item()
        loss_abs_diff = abs(loss_cp - loss_eager)
        print(f"\n{'=' * 72}")
        print(f"MiniMax M3 FULL-MODEL CP forward equivalence  cp_size={cp_size}  seqlen={seqlen}  packed={packed}")
        print(f"{'=' * 72}")
        print(f"logits mean_abs={mean_abs:.3e}  max_abs={max_abs:.3e}  top1_agreement={top1:.4f}")
        print(f"loss eager={loss_eager:.8f}  cp={loss_cp:.8f}  abs_diff={loss_abs_diff:.3e}")
        print(f"worst positions (per-token logit drift): {worst}")
        # fp32: tight (only kernel rounding). bf16: looser; top1 (argmax) is the
        # leakage-robust signal -- a causality/sharding bug tanks top1 + concentrates
        # drift at early positions; bf16 reduction-order noise scatters and keeps top1 high.
        if fp32:
            ok = top1 > 0.999 and mean_abs < 5e-3 and loss_abs_diff < 1e-4
        else:
            ok = top1 > 0.98 and mean_abs < 1e-1 and loss_abs_diff < 1e-3
        print("RESULT:", "PASS (full-model CP == eager)" if ok else "FAIL (forward diverges under real CP sharding)")
        print(f"{'=' * 72}\n")
        rc = 0 if ok else 1
    else:
        rc = 0

    rc_t = torch.tensor(rc, device=device)
    dist.all_reduce(rc_t, op=dist.ReduceOp.MAX)
    dist.barrier()
    dist.destroy_process_group()
    sys.exit(int(rc_t.item()))


if __name__ == "__main__":
    main()
