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

"""Identical document and prefix masks for the module benchmark only."""

from collections.abc import Callable
from itertools import accumulate
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

if TYPE_CHECKING:
    from magi_attention.dist_attn_runtime_mgr import DistAttnRuntimeKey
    from transformer_engine.pytorch.attention import DotProductAttention

Attention = Callable[..., torch.Tensor]


def boundaries(seq: int, documents: int = 4) -> list[int]:
    """Return four or sixteen unequal document boundaries for a fixed token count."""
    lengths = [seq // (2 * documents) * n for n in [1, 1, 2, 4] * (documents // 4)]
    return [0] + list(accumulate(lengths))


def column_mask(seq: int, kind: str, documents: int = 4) -> torch.Tensor:
    """Encode allowed query rows [upper, lower) for every key column."""
    mask = torch.empty((1, 1, seq, 2), device="cuda", dtype=torch.int32)
    for start, end in zip(boundaries(seq, documents)[:-1], boundaries(seq, documents)[1:]):
        mask[:, :, start:end, 0] = end
        mask[:, :, start:end, 1] = torch.arange(start, end, device="cuda", dtype=torch.int32)
        if kind == "prefix":
            mask[:, :, start : (start + end) // 2, 1] = start
    return mask


def reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, kind: str, documents: int = 4) -> torch.Tensor:
    """Compute BSHD attention independently using per-document PyTorch SDPA.

    Prefix queries attend bidirectionally within their prefix. Suffix queries
    attend all previous positions within their document. No cross-document keys.
    """
    outputs = []
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        for start, end in zip(boundaries(q.shape[1], documents)[:-1], boundaries(q.shape[1], documents)[1:]):
            qs, ks, vs = (t[:, start:end].transpose(1, 2) for t in (q, k, v))
            causal = F.scaled_dot_product_attention(qs, ks, vs, is_causal=True, enable_gqa=True)
            if kind == "prefix":
                half = (end - start) // 2
                prefix = F.scaled_dot_product_attention(
                    qs[:, :, :half], ks[:, :, :half], vs[:, :, :half], enable_gqa=True
                )
                causal = torch.cat((prefix, causal[:, :, half:]), dim=2)
            outputs.append(causal.transpose(1, 2))
    return torch.cat(outputs, dim=1)


def magi_key(seq: int, kind: str, group: torch.distributed.ProcessGroup, documents: int = 4) -> "DistAttnRuntimeKey":
    """Construct native Magi ranges; its dispatcher owns the CP token layout."""
    from magi_attention.api import magi_attn_flex_key
    from magi_attention.common import AttnRanges

    q_ranges, k_ranges, types = [], [], []
    for start, end in zip(boundaries(seq, documents)[:-1], boundaries(seq, documents)[1:]):
        if kind == "prefix":
            mid = (start + end) // 2
            q_ranges.extend([[start, mid], [mid, end]])
            k_ranges.extend([[start, mid], [start, end]])
            types.extend(["full", "causal"])
        else:
            q_ranges.append([start, end])
            k_ranges.append([start, end])
            types.append("causal")
    return magi_attn_flex_key(
        q_ranges=AttnRanges.from_ranges(q_ranges),
        k_ranges=AttnRanges.from_ranges(k_ranges),
        attn_mask_type=types,
        total_seqlen_q=seq,
        total_seqlen_k=seq,
        num_heads_q=32,
        num_heads_kv=4,
        head_dim=128,
        pad_size=0,
        cp_group_or_mesh=group,
        chunk_size=512,
    )


def fa4_mask(seq: int, kind: str, pack_gqa: bool = True) -> Attention:
    """Build FA4 mask_mod and forward/backward block-sparse metadata once.

    Use the installed FA4 public autograd API and the sparse tensor convention
    from upstream tests/cute/test_mask_mod.py. Construction is outside timing.
    """
    import cutlass
    import cutlass.cute as cute
    from flash_attn.cute.block_sparsity import BlockSparseTensorsTorch
    from flash_attn.cute.interface import flash_attn_func
    from torch.nn.attention.flex_attention import create_block_mask

    @cute.jit
    def mask_mod(batch, head, m, n, seqlen_info, aux_tensors):
        same_doc = (
            ((m < seq // 8) & (n < seq // 8))
            | ((m >= seq // 8) & (m < seq // 4) & (n >= seq // 8) & (n < seq // 4))
            | ((m >= seq // 4) & (m < seq // 2) & (n >= seq // 4) & (n < seq // 2))
            | ((m >= seq // 2) & (n >= seq // 2))
        )
        prefix_key = (
            (n < seq // 16)
            | ((n >= seq // 8) & (n < 3 * seq // 16))
            | ((n >= seq // 4) & (n < 3 * seq // 8))
            | ((n >= seq // 2) & (n < 3 * seq // 4))
        )
        if cutlass.const_expr(kind == "prefix"):
            return same_doc & ((m >= n) | prefix_key)
        return same_doc & (m >= n)

    def flex_mask(b, h, m, n):
        same_doc = torch.zeros_like(m + n, dtype=torch.bool)
        prefix_key = torch.zeros_like(n, dtype=torch.bool)
        for start, end in zip(boundaries(seq)[:-1], boundaries(seq)[1:]):
            same_doc = same_doc | ((m >= start) & (m < end) & (n >= start) & (n < end))
            prefix_key = prefix_key | ((n >= start) & (n < (start + end) // 2))
        return same_doc & ((m >= n) | prefix_key) if kind == "prefix" else same_doc & (m >= n)

    tile = (256, 128) if torch.cuda.get_device_capability()[0] == 10 else (128, 128)
    bm = create_block_mask(flex_mask, 1, 1, seq, seq, device="cuda", BLOCK_SIZE=tile)
    _, _, kc, ki, fk, fki, qc, qi, fq, fqi, *_ = bm.as_tuple()
    fwd = BlockSparseTensorsTorch(kc, ki, fk, fki, block_size=tile)
    bwd = BlockSparseTensorsTorch(qc, qi, fq, fqi, block_size=tile)

    def attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        causal: bool = True,
        window_size: tuple[int | None, int | None] = (None, None),
    ) -> torch.Tensor:
        output, _ = flash_attn_func(
            q,
            k,
            v,
            causal=False,
            mask_mod=mask_mod,
            block_sparse_tensors=fwd,
            block_sparse_tensors_bwd=bwd,
            pack_gqa=pack_gqa,
        )
        return output

    return attention


def varlen_attention(seq: int, kind: str, backend: str, te_attention: "DotProductAttention | None" = None) -> Attention:
    """Use one native varlen call for documents, two disjoint Q calls for prefix.

    The prefix decomposition duplicates no attention pairs. Packing Q/K/V and
    restoring output order are deliberately inside module timing.
    """
    bounds = boundaries(seq)
    lengths = [b - a for a, b in zip(bounds[:-1], bounds[1:])]
    cu = torch.tensor(bounds, dtype=torch.int32, device="cuda")
    half_cu = cu // 2
    prefix_idx = torch.cat([torch.arange(a, (a + b) // 2, device="cuda") for a, b in zip(bounds[:-1], bounds[1:])])
    suffix_idx = torch.cat([torch.arange((a + b) // 2, b, device="cuda") for a, b in zip(bounds[:-1], bounds[1:])])
    restore = torch.argsort(torch.cat((prefix_idx, suffix_idx)))
    if backend == "fa4":
        from flash_attn.cute.interface import flash_attn_varlen_func
    elif backend == "fa3":
        from flash_attn_interface import flash_attn_varlen_func

    def call(q, k, v, cq, ck, mq, mk, causal):
        if backend == "te":
            assert te_attention is not None
            return te_attention(
                q,
                k,
                v,
                qkv_format="thd",
                attn_mask_type="padding_causal_bottom_right" if causal else "padding",
                cu_seqlens_q=cq,
                cu_seqlens_kv=ck,
                max_seqlen_q=mq,
                max_seqlen_kv=mk,
                window_size=(-1, 0) if causal else (-1, -1),
            ).reshape(q.shape[0], 32, 128)
        output = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cq,
            cu_seqlens_k=ck,
            max_seqlen_q=mq,
            max_seqlen_k=mk,
            causal=causal,
        )
        return output[0] if backend == "fa4" else output

    def attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        causal: bool = True,
        window_size: tuple[int | None, int | None] = (None, None),
    ) -> torch.Tensor:
        q, k, v = (t.squeeze(0) for t in (q, k, v))
        if kind == "documents":
            return call(q, k, v, cu, cu, max(lengths), max(lengths), True).unsqueeze(0)
        qp, kp, vp = (t.index_select(0, prefix_idx) for t in (q, k, v))
        qs = q.index_select(0, suffix_idx)
        prefix = call(qp, kp, vp, half_cu, half_cu, max(lengths) // 2, max(lengths) // 2, False)
        suffix = call(qs, k, v, half_cu, cu, max(lengths) // 2, max(lengths), True)
        return torch.cat((prefix, suffix)).index_select(0, restore).unsqueeze(0)

    return attention
