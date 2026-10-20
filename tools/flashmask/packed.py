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

"""THD module benchmark adapters; these do not enable production recipe packing."""

from collections.abc import Callable
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
from torch.distributed.nn.functional import all_to_all_single

from nemo_automodel.components.attention.flashmask_attention import FlashMaskAttention
from nemo_automodel.components.models.qwen3_moe.layers import Qwen3MoeAttention

if TYPE_CHECKING:
    from magi_attention.dist_attn_runtime_mgr import DistAttnRuntimeKey


def sequence_to_heads(x: torch.Tensor, group: dist.ProcessGroup, order: torch.Tensor) -> torch.Tensor:
    """Exchange THD sequence shards and restore global document-token order."""
    size = group.size()
    tokens, heads, channels = x.shape
    send = x.reshape(tokens, size, heads // size, channels).permute(1, 0, 2, 3).contiguous()
    recv = all_to_all_single(torch.empty_like(send), send, group=group)
    return recv.flatten(0, 1).index_select(0, order)


def heads_to_sequence(x: torch.Tensor, group: dist.ProcessGroup, inverse: torch.Tensor) -> torch.Tensor:
    """Apply the inverse packed-token permutation and exchange head shards."""
    size = group.size()
    tokens, heads, channels = x.shape
    send = x.index_select(0, inverse).reshape(size, tokens // size, heads, channels)
    recv = all_to_all_single(torch.empty_like(send), send, group=group)
    return recv.permute(1, 0, 2, 3).reshape(tokens // size, size * heads, channels)


def bind_packed_attention(
    module: Qwen3MoeAttention,
    backend: str,
    cu: torch.Tensor,
    max_seqlen: int,
    group: dist.ProcessGroup,
    column_mask: torch.Tensor,
    magi_runtime_key: "DistAttnRuntimeKey | None",
    per_rank_indices: list[torch.Tensor],
) -> Callable[..., torch.Tensor]:
    """Bind THD kernel calls while retaining the complete native Qwen forward.

    TE and Magi consume native THD. FA3/FA4 call native varlen kernels. FlashMask
    takes a no-copy leading batch view; its packed Ulysses transport uses the
    same per-document partition as TE, with differentiable token permutations.
    The latter is a benchmark candidate, not the production BSHD-only adapter.
    """
    attention = module.attn_module
    prior = module.attn_func
    size = group.size()
    if backend == "te":
        return attention
    if backend == "magi":
        from magi_attention.api import calc_attn

        assert magi_runtime_key is not None

        def magi(
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            **kwargs: torch.Tensor | int | str | bool | tuple[int, int],
        ) -> torch.Tensor:
            out, _ = calc_attn(q, k, v, key=magi_runtime_key)
            return out

        return magi
    # TE's preprocessing leaves THD Q/K/V unchanged and forwards their metadata.
    # Only the preprocessing selector changes; the module's projections, norms,
    # RoPE, actual attention callable and output projection stay explicit here.
    module.backend.attn = "te"
    if backend in ("fa3", "fa4"):
        if backend == "fa4":
            from flash_attn.cute.interface import flash_attn_varlen_func
        else:
            from flash_attn_interface import flash_attn_varlen_func

        def fa(
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            **kwargs: torch.Tensor | int | str | bool | tuple[int, int],
        ) -> torch.Tensor:
            out = flash_attn_varlen_func(
                q,
                k,
                v,
                cu_seqlens_q=cu,
                cu_seqlens_k=cu,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                causal=True,
            )
            return out[0] if backend == "fa4" else out

        return fa
    if backend == "flashmask":
        assert isinstance(attention, FlashMaskAttention)
        attention.cp_group = None
        attention.causal = False
        inverse = torch.cat(per_rank_indices).long()
        order = torch.argsort(inverse)

        def flashmask(
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            **kwargs: torch.Tensor | int | str | bool | tuple[int, int],
        ) -> torch.Tensor:
            if size > 1:
                q, k, v = (sequence_to_heads(t, group, order) for t in (q, k, v))
            out = attention(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), startend_row_indices=column_mask).squeeze(0)
            return heads_to_sequence(out, group, inverse) if size > 1 else out

        return flashmask

    def upstream(
        q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, **kwargs: torch.Tensor | int | str | bool | tuple[int, int]
    ) -> torch.Tensor:
        return prior(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)).squeeze(0)

    return upstream
