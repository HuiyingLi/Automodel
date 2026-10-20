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

"""FlashMask-4 attention with Ulysses context parallelism."""

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.nn.functional import all_to_all_single

from nemo_automodel.shared.import_utils import safe_import


def _sequence_to_heads(x: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    """Exchange a head-tail sequence shard for a contiguous head shard.

    Args:
        x: Tensor of shape [batch, local_sequence, heads, channels], in the
            two-chunk load-balanced CP layout. Sequence must be even and heads
            divisible by the group size. CUDA BF16 or FP16.
        group: Context-parallel process group.

    Returns:
        Tensor of shape [batch, global_sequence, heads / cp_size, channels]
        in global token order. The result does not alias the input.
    """
    size = group.size()
    b, s, h, d = x.shape
    send = x.reshape(b, s, size, h // size, d).permute(2, 0, 1, 3, 4).contiguous()
    recv = all_to_all_single(torch.empty_like(send), send, group=group)
    chunks = recv.reshape(size, b, 2, s // 2, h // size, d).permute(1, 0, 2, 3, 4, 5)
    ordered = torch.cat((chunks[:, :, 0], chunks[:, :, 1].flip(1)), dim=1)
    return ordered.reshape(b, size * s, h // size, d)


def _heads_to_sequence(x: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    """Restore the head-tail sequence partition after attention.

    Args:
        x: Tensor of shape [batch, global_sequence, local_heads, channels],
            with global tokens in sequence order and a contiguous head shard.
        group: Context-parallel process group; global_sequence is divisible
            by twice the group size.

    Returns:
        Tensor of shape [batch, global_sequence / cp_size, heads, channels]
        in the head-tail CP layout, independently allocated.
    """
    size = group.size()
    b, s, h, d = x.shape
    chunks = x.reshape(b, 2 * size, s // (2 * size), h, d)
    paired = torch.stack((chunks[:, :size], chunks[:, size:].flip(1)), dim=2)
    send = paired.reshape(b, size, s // size, h, d).transpose(0, 1).contiguous()
    recv = all_to_all_single(torch.empty_like(send), send, group=group)
    return recv.permute(1, 2, 0, 3, 4).reshape(b, s // size, size * h, d)


class FlashMaskAttention(nn.Module):
    """Optional FlashMask-4 backend for BSHD attention.

    CP uses differentiable Ulysses all-to-all on both Hopper and Blackwell.
    It consumes the framework's head-tail sequence partition and requires CP
    to divide both Q and KV head counts. This is distinct from upstream's
    SM100-only NCCL-GIN overlap implementation.
    """

    def __init__(self, *, softmax_scale: float, causal: bool = True) -> None:
        super().__init__()
        available, frontend = safe_import("flashmask.frontend.torch")
        if not available:
            raise ImportError("attn='flashmask' requires FlashMask's PyTorch frontend; see tools/flashmask/README.md")
        self._attention = frontend.flashmask_attention
        self.softmax_scale = softmax_scale
        self.causal = causal
        self.cp_group: dist.ProcessGroup | None = None

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        startend_row_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run FlashMask, including CP exchanges when configured.

        Args:
            q: CUDA FP16/BF16 tensor of shape [batch, sequence, heads, channels].
            k: Tensor of shape [batch, sequence, kv_heads, channels], same dtype
                and device as q.
            v: Tensor of shape [batch, sequence, kv_heads, value_channels], same
                dtype and device as q. Under CP all three sequences are local
                head-tail shards, with an even local sequence length.
            startend_row_indices: Optional CUDA int32 column mask of shape
                [batch, 1, global_sequence, bounds], in global token order.
                Bounds follow FlashMask's public column-mask contract (1/2
                for causal, 2/4 for noncausal). Only broadcast masks are
                supported under CP; dense causal attention needs no mask.

        Returns:
            Tensor of shape [batch, sequence, heads, value_channels] with the
            same sequence partition as q; no input is mutated.
        """
        if any(t.ndim != 4 for t in (q, k, v)):
            raise ValueError("FlashMask requires BSHD tensors")
        group = self.cp_group
        if group is not None and group.size() > 1:
            size = group.size()
            if q.shape[1] % 2 or any(t.shape[2] % size for t in (q, k, v)):
                raise ValueError("FlashMask CP requires even local sequence and CP dividing Q/K/V head counts")
            if startend_row_indices is not None and startend_row_indices.shape[1] != 1:
                raise ValueError("FlashMask CP requires a head-broadcast column mask")
            q, k, v = (_sequence_to_heads(t, group) for t in (q, k, v))
        out = self._attention(
            q,
            k,
            v,
            startend_row_indices=startend_row_indices,
            causal=self.causal,
            softmax_scale=self.softmax_scale,
        )
        if group is not None and group.size() > 1:
            out = _heads_to_sequence(out, group)
        return out
