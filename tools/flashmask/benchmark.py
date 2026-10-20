# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Qwen3-30B-A3B attention benchmark; launch with torchrun, one CP group.

Times the native attention backend and the complete Qwen attention module.
Synthetic, seeded inputs exercise full 4K/8K causal sequences; no model
checkpoint is loaded. CP communication is included, parameter-gradient
synchronization and input partitioning are outside module timing.
"""

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import os
import subprocess
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import Qwen3MoeConfig

from nemo_automodel.components.attention.utils import (
    initialize_attn_module_and_func,
    postprocess_output_for_attn,
    preprocess_args_and_kwargs_for_attn,
)
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.qwen3_moe.layers import Qwen3MoeAttention


def _metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    """Compare same-layout tensors, reducing all their elements.

    Args:
        actual: Tensor of any shape, the tested output or gradient.
        expected: Reference tensor of exactly the same shape.

    Returns:
        Maximum absolute error, normalized maximum and relative RMS error.
    """
    a, b = actual.float(), expected.float()
    delta = a - b
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise AssertionError("Non-finite output/gradient")
    return {
        "max_abs": delta.abs().max().item(),
        "relative_max": (delta.abs().max() / b.abs().max().clamp_min(1e-9)).item(),
        "relative_rms": (delta.square().mean().sqrt() / b.square().mean().sqrt().clamp_min(1e-9)).item(),
    }


def _config() -> Qwen3MoeConfig:
    return Qwen3MoeConfig(
        hidden_size=2048,
        num_attention_heads=32,
        num_key_value_heads=4,
        head_dim=128,
        attention_dropout=0.0,
        attention_bias=False,
        rms_norm_eps=1e-6,
        rope_theta=1000000.0,
        torch_dtype=torch.bfloat16,
    )


def _core_reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Run the PyTorch Flash SDPA reference over full BSHD Q/K/V tensors.

    Args:
        q: Tensor of shape [1, global_sequence, 32, 128].
        k: Tensor of shape [1, global_sequence, 4, 128].
        v: Tensor with the same shape as k.

    Returns:
        Tensor of shape [1, global_sequence, 32, 128].
    """
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        return F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True, enable_gqa=True
        ).transpose(1, 2)


def main() -> None:
    """Validate and time the requested native attention backend on one CP group."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend", choices=("flashmask", "flashmask_overlap", "te", "fa3", "fa4", "sdpa", "magi"), required=True
    )
    parser.add_argument("--flashmask-source", type=Path, help="Pinned upstream checkout for the overlap comparison")
    parser.add_argument("--seq", type=int, nargs="+", default=[4096, 8192])
    parser.add_argument("--scope", nargs="+", choices=("core", "module"), default=["core", "module"])
    parser.add_argument("--execution", nargs="+", choices=("eager", "graph"), default=["eager"])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", device_id=torch.device("cuda", torch.cuda.current_device()))
    size, rank = dist.get_world_size(), dist.get_rank()
    mesh = init_device_mesh("cuda", (size,), mesh_dim_names=("cp",))
    group = mesh.get_group()
    if args.backend in ("fa3", "fa4") and size > 1:
        raise ValueError("Native AutoModel FA4 has no CP path; use sdpa for the PyTorch FlashAttention ring baseline")
    repo = Path(__file__).resolve().parents[2]
    metadata = {
        "backend": args.backend,
        "cp": size,
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "nccl": dist.get_backend(group),
        "source_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "source_diff_sha256": hashlib.sha256(subprocess.check_output(["git", "diff", "HEAD"], cwd=repo)).hexdigest(),
        "python": os.sys.executable,
        "seed": 4397,
        "dtype": "bfloat16",
        "batch": 1,
        "model": "Qwen/Qwen3-30B-A3B",
        "q_heads": 32,
        "kv_heads": 4,
        "head_dim": 128,
        "synthetic_weights": True,
        "packing": False,
        "execution": args.execution,
        "cp_algorithm": {
            "flashmask": "ulysses",
            "flashmask_overlap": "upstream_nccl_gin_overlap",
            "te": "p2p",
            "sdpa": "torch_ring",
            "magi": "magi",
            "fa4": "none",
            "fa3": "none",
        }[args.backend]
        if size > 1
        else "none",
        "packages": {
            n: importlib.metadata.version(n)
            for n in ("transformers", "transformer-engine", "nvidia-cutlass-dsl", "flash-attn-4", "magi_attention")
        },
    }
    metadata["magi_kernel_backend"] = os.environ.get("MAGI_ATTENTION_KERNEL_BACKEND", "ffa")
    native_backend = {"fa3": "fa4", "flashmask_overlap": "flashmask"}.get(args.backend, args.backend)
    if args.backend == "flashmask_overlap":
        if args.flashmask_source is None or size == 1:
            raise ValueError("The upstream overlap comparison requires --flashmask-source and CP > 1")
        os.sys.path.insert(0, str(args.flashmask_source / "tests/torch/distributed"))
        from overlap_flashmask import overlap_flashmask_attention_fm4

        metadata["flashmask_source_sha"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=args.flashmask_source, text=True
        ).strip()
    if args.backend == "fa3":
        from flash_attn_interface import flash_attn_func

        def fa3(
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            *,
            causal: bool = True,
            window_size: tuple[int | None, int | None] = (None, None),
        ) -> torch.Tensor:
            """Call installed FA3 with BSHD tensors, returning a BSHD tensor.

            Args:
                q: Tensor of shape [batch, sequence, query_heads, head_dim].
                k: Tensor of shape [batch, sequence, kv_heads, head_dim].
                v: Tensor with the same shape as k.
                causal: Apply a causal mask.
                window_size: Left and right attention distances, None for unbounded.

            Returns:
                Tensor of shape [batch, sequence, query_heads, head_dim].
            """
            return flash_attn_func(
                q, k, v, causal=causal, window_size=tuple(-1 if t is None else t for t in window_size)
            )

    results = []
    for seq in args.seq:
        for scope in args.scope:
            torch.manual_seed(4397)
            if seq % (2 * size):
                raise ValueError("Sequence must divide evenly into 2 * CP chunks")
            key = None
            if args.backend == "magi":
                from magi_attention.api import dispatch, magi_attn_varlen_key

                from nemo_automodel.components.distributed.context_parallel.magi import set_active_cp_group

                set_active_cp_group(group)
                cu = torch.tensor([0, seq], device="cuda", dtype=torch.int32)
                key = magi_attn_varlen_key(
                    cu_seqlens_q=cu,
                    cu_seqlens_k=cu,
                    num_heads_q=32,
                    num_heads_kv=4,
                    head_dim=128,
                    pad_size=0,
                    cp_group_or_mesh=group,
                    causal=True,
                    chunk_size=512,
                )
            chunk = seq // (2 * size)
            idx = torch.cat(
                (
                    torch.arange(rank * chunk, (rank + 1) * chunk, device="cuda"),
                    torch.arange((2 * size - 1 - rank) * chunk, (2 * size - rank) * chunk, device="cuda"),
                )
            )

            def partition(tensor: torch.Tensor) -> torch.Tensor:
                """Partition global [1, sequence, ...] tensors identically to attention.

                Args:
                    tensor: Tensor of shape [1, global_sequence, ...].

                Returns:
                    Tensor of shape [1, local_sequence, ...], detached from
                    its source and using this backend's token ordering.
                """
                if key is not None:
                    return dispatch(tensor.squeeze(0), key=key).unsqueeze(0).detach().contiguous()
                return tensor.index_select(1, idx).detach().contiguous()

            cp_dummy = torch.empty(seq, device="cuda")

            def cp_context():
                if args.backend == "sdpa" and size > 1:
                    from torch.distributed.tensor.experimental import context_parallel

                    return context_parallel(mesh, buffers=[cp_dummy], buffer_seq_dims=[0])
                return contextlib.nullcontext()

            torch.manual_seed(4397)
            ref_module = None
            if scope == "module":
                cfg = _config()
                ref_module = Qwen3MoeAttention(
                    cfg, BackendConfig(attn="sdpa", linear="torch", rms_norm="torch", rope_fusion=False)
                ).cuda()
                module = Qwen3MoeAttention(
                    cfg, BackendConfig(attn=native_backend, linear="torch", rms_norm="torch", rope_fusion=False)
                ).cuda()
                state = module.state_dict()
                reference_state = ref_module.state_dict()
                if {k for k in state if not k.endswith("_extra_state")} != set(reference_state):
                    raise AssertionError("Reference and tested module parameter/buffer keys differ")
                state.update(reference_state)
                module.load_state_dict(state, strict=True)
                torch.manual_seed(4398)
                full = [torch.randn(1, seq, 2048, device="cuda", dtype=torch.bfloat16, requires_grad=True)]
                positions = torch.arange(seq, device="cuda", dtype=torch.float32)
                inv_freq = 1000000.0 ** (-torch.arange(0, 128, 2, device="cuda", dtype=torch.float32) / 128)
                angles = positions[:, None] * inv_freq[None, :]
                freqs = torch.cat((angles.cos(), angles.sin()), -1).unsqueeze(0)
                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    expected = ref_module(full[0], freqs_cis=freqs)
                freqs_local = partition(freqs)
                attention = module.attn_module
            else:
                attention, attn_func = initialize_attn_module_and_func(
                    native_backend, 32, 128, 128, 128**-0.5, num_gqa_groups=4
                )
                module = attention
                torch.manual_seed(4398)
                full = [
                    torch.randn(1, seq, h, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
                    for h in (32, 4, 4)
                ]
                expected = _core_reference(*full)
            if args.backend == "flashmask_overlap":
                # Noncausal column bounds encode the identical global causal
                # mask. The upstream wrapper rebases it for head-tail CP.
                column_mask = torch.empty(1, 1, seq, 2, device="cuda", dtype=torch.int32)
                column_mask[..., 0] = seq
                column_mask[..., 1] = torch.arange(seq, device="cuda", dtype=torch.int32)

                def overlap(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
                    """Run upstream's single-node CP overlap wrapper on local BSHD inputs.

                    Args:
                        q: Local query tensor in head-tail sequence order.
                        k: Local key tensor in the same sequence order.
                        v: Local value tensor in the same sequence order.

                    Returns:
                        Local attention output with q's sequence/head layout.
                    """
                    return overlap_flashmask_attention_fm4(q, k, v, column_mask, causal=False, group=group)

                if scope == "module":
                    module.attn_func = overlap
                else:
                    attn_func = overlap
            if args.backend == "fa3":
                if scope == "module":
                    module.attn_func = fa3
                else:
                    attn_func = fa3
            dout = torch.randn_like(expected)
            expected.backward(dout)
            expected_local, dout_local = partition(expected.detach()), partition(dout)
            input_reference_grads = [partition(t.grad) for t in full]
            local = [partition(t).requires_grad_() for t in full]
            if args.backend == "te" and size > 1:
                attention.set_context_parallel_group(
                    group, dist.get_process_group_ranks(group), torch.cuda.Stream(), cp_comm_type="p2p"
                )
            if args.backend == "flashmask":
                if scope == "module":
                    module.setup_cp_attention(mesh)
                else:
                    attention.cp_group = group

            def forward() -> torch.Tensor:
                """Run a native backend on the captured local input tensors.

                Returns:
                    Local BSHD core output or [1, local_sequence, 2048] module output.
                """
                if scope == "module":
                    return module(local[0], freqs_cis=freqs_local)
                q, k, v, kw = preprocess_args_and_kwargs_for_attn(*local, None, native_backend)
                out = postprocess_output_for_attn(attn_func(q, k, v, **kw), native_backend)
                return out.reshape(1, local[0].shape[1], 32, 128)

            def reset_grads() -> None:
                for t in local:
                    t.grad = None
                if module is not None:
                    module.zero_grad(set_to_none=True)

            with cp_context():
                actual = forward()
                actual.backward(dout_local)
            checks = {"output": _metrics(actual, expected_local)}
            for i, (t, ref) in enumerate(zip(local, input_reference_grads)):
                checks[f"input_grad_{i}"] = _metrics(t.grad, ref)
            if ref_module is not None:
                for (name, param), (_, ref) in zip(module.named_parameters(), ref_module.named_parameters()):
                    grad = param.grad.float()
                    dist.all_reduce(grad, group=group)
                    checks[name] = _metrics(grad, ref.grad)
            all_checks = [None] * size
            dist.all_gather_object(all_checks, checks)
            if any(m["relative_rms"] > 0.02 or m["relative_max"] > 0.03 for c in all_checks for m in c.values()):
                if rank == 0:
                    print(json.dumps({"failed_parity": all_checks, "scope": scope, "seq": seq}), flush=True)
                raise AssertionError("Output/gradient parity exceeded 2% relative RMS or 3% normalized max")
            data_hash = hashlib.sha256(full[0].detach().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()
            if rank == 0:
                print(
                    json.dumps(
                        {
                            "parity_pass": True,
                            "backend": args.backend,
                            "cp": size,
                            "seq": seq,
                            "scope": scope,
                            "worst_relative_rms": max(m["relative_rms"] for c in all_checks for m in c.values()),
                        }
                    ),
                    flush=True,
                )
            del expected, actual, ref_module, full
            torch.cuda.empty_cache()
            for mode in ("forward", "forward_backward"):

                def step() -> torch.Tensor:
                    """Run one local forward, optionally backward; return its output tensor."""
                    with cp_context():
                        out = forward()
                        if mode == "forward_backward":
                            out.backward(dout_local)
                    return out

                reset_grads()
                for _ in range(args.warmup):
                    step()
                    reset_grads()
                torch.cuda.synchronize()
                trial_results = []
                for trial in range(args.trials):
                    reset_grads()
                    dist.barrier()
                    torch.cuda.reset_peak_memory_stats()
                    begins = [torch.cuda.Event(enable_timing=True) for _ in range(args.iterations)]
                    ends = [torch.cuda.Event(enable_timing=True) for _ in range(args.iterations)]
                    wall = time.perf_counter()
                    for i in range(args.iterations):
                        reset_grads()
                        begins[i].record()
                        step()
                        ends[i].record()
                    torch.cuda.synchronize()
                    wall_ms = (time.perf_counter() - wall) * 1000 / args.iterations
                    samples = torch.tensor([a.elapsed_time(b) for a, b in zip(begins, ends)], device="cuda")
                    dist.all_reduce(samples, op=dist.ReduceOp.MAX)
                    wall_tensor = torch.tensor(wall_ms, device="cuda")
                    dist.all_reduce(wall_tensor, op=dist.ReduceOp.MAX)
                    trial_results.append(
                        {
                            "median_ms": samples.median().item(),
                            "p20_ms": samples.quantile(0.2).item(),
                            "p80_ms": samples.quantile(0.8).item(),
                            "mean_ms": samples.mean().item(),
                            "wall_ms": wall_tensor.item(),
                            "peak_memory_bytes": torch.cuda.max_memory_allocated(),
                            "samples_ms": samples.cpu().tolist(),
                        }
                    )
                row = {
                    "execution": "eager",
                    "seq": seq,
                    "scope": scope,
                    "mode": mode,
                    "input_sha256": data_hash,
                    "parity": all_checks,
                    "trials": trial_results,
                }
                results.append(row)
                if rank == 0:
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    args.output.write_text(json.dumps({"metadata": metadata, "results": results}, indent=2))
                    print(
                        json.dumps(
                            {
                                "backend": args.backend,
                                "cp": size,
                                "seq": seq,
                                "scope": scope,
                                "mode": mode,
                                "median_ms": [t["median_ms"] for t in trial_results],
                            }
                        ),
                        flush=True,
                    )
                if "graph" in args.execution:
                    reset_grads()
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        for _ in range(3):
                            step()
                            reset_grads()
                    torch.cuda.current_stream().wait_stream(stream)
                    dist.barrier()
                    graph = torch.cuda.CUDAGraph()
                    repeats = 5
                    with torch.cuda.graph(graph, stream=stream):
                        for _ in range(repeats):
                            reset_grads()
                            captured_output = step()
                    for _ in range(args.warmup):
                        graph.replay()
                    torch.cuda.synchronize()
                    graph_checks = {"output": _metrics(captured_output, expected_local)}
                    if mode == "forward_backward":
                        graph_checks.update(
                            {
                                f"input_grad_{i}": _metrics(t.grad, ref)
                                for i, (t, ref) in enumerate(zip(local, input_reference_grads))
                            }
                        )
                    if any(m["relative_rms"] > 0.02 or m["relative_max"] > 0.03 for m in graph_checks.values()):
                        raise AssertionError(f"CUDA Graph replay parity failed: {graph_checks}")
                    graph_trials = []
                    for _ in range(args.trials):
                        dist.barrier()
                        starts = [torch.cuda.Event(enable_timing=True) for _ in range(args.iterations)]
                        finishes = [torch.cuda.Event(enable_timing=True) for _ in range(args.iterations)]
                        for i in range(args.iterations):
                            starts[i].record()
                            graph.replay()
                            finishes[i].record()
                        torch.cuda.synchronize()
                        times = torch.tensor(
                            [a.elapsed_time(b) / repeats for a, b in zip(starts, finishes)], device="cuda"
                        )
                        dist.all_reduce(times, op=dist.ReduceOp.MAX)
                        graph_trials.append(
                            {
                                "median_ms": times.median().item(),
                                "p20_ms": times.quantile(0.2).item(),
                                "p80_ms": times.quantile(0.8).item(),
                                "mean_ms": times.mean().item(),
                                "samples_ms": times.cpu().tolist(),
                            }
                        )
                    results.append(
                        {
                            "execution": "graph",
                            "seq": seq,
                            "scope": scope,
                            "mode": mode,
                            "input_sha256": data_hash,
                            "parity": all_checks,
                            "trials": graph_trials,
                            "steps_per_replay": repeats,
                        }
                    )
                    if rank == 0:
                        args.output.write_text(json.dumps({"metadata": metadata, "results": results}, indent=2))
                        print(
                            json.dumps(
                                {
                                    "backend": args.backend,
                                    "cp": size,
                                    "seq": seq,
                                    "scope": scope,
                                    "mode": mode,
                                    "execution": "graph",
                                    "median_ms": [t["median_ms"] for t in graph_trials],
                                }
                            ),
                            flush=True,
                        )
                    # NCCL communicators captured by a live graph cannot be
                    # destroyed during process-group teardown.
                    graph.reset()
                    del graph, captured_output, stream
            module = None
            local = []
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
