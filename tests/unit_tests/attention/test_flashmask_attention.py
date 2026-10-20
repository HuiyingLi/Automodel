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

"""CPU contracts and distributed gradient checks for FlashMask integration."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import nemo_automodel.components.attention.flashmask_attention as fm
from nemo_automodel.components.attention.utils import (
    initialize_attn_module_and_func,
    preprocess_args_and_kwargs_for_attn,
)


def _math_attention(q, k, v, *, causal, softmax_scale, startend_row_indices=None):
    """Independent FP32 dense oracle, with no FlashMask kernel dependency.

    Args:
        q: Tensor of shape [batch, sequence, query_heads, channels].
        k: Tensor of shape [batch, sequence, kv_heads, channels].
        v: Tensor with the same shape as k.
        causal: Whether keys after each query are masked.
        softmax_scale: Scale applied before softmax.
        startend_row_indices: Unused; these transport tests use dense causal attention.

    Returns:
        Tensor of shape [batch, sequence, query_heads, channels].
    """
    assert startend_row_indices is None
    qh = q.transpose(1, 2)
    kh = k.transpose(1, 2).repeat_interleave(q.shape[2] // k.shape[2], 1)
    vh = v.transpose(1, 2).repeat_interleave(q.shape[2] // v.shape[2], 1)
    scores = (qh @ kh.transpose(-1, -2)) * softmax_scale
    if causal:
        allowed = torch.arange(q.shape[1])[:, None] >= torch.arange(k.shape[1])[None, :]
        scores = scores.masked_fill(~allowed, -torch.inf)
    return (scores.softmax(-1) @ vh).transpose(1, 2)


def _worker(rank: int, rendezvous: str, wrong_order: bool) -> None:
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        torch.manual_seed(4397)
        full = [torch.randn(1, 12, h, 4, requires_grad=True) for h in (4, 2, 2)]
        upstream = torch.randn_like(full[0])
        ref = _math_attention(*full, causal=True, softmax_scale=.5)
        ref.backward(upstream)
        indices = torch.tensor(list(range(rank * 3, (rank + 1) * 3)) +
                               list(range((3 - rank) * 3, (4 - rank) * 3)))
        local = [x.detach().index_select(1, indices).requires_grad_() for x in full]
        # Replace only the unavailable CUDA kernel. Real Gloo all-to-all and
        # its autograd run; the expected result is independent dense math.
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(fm, "safe_import", lambda name: (True, SimpleNamespace(flashmask_attention=_math_attention)))
            if wrong_order:
                original = fm._sequence_to_heads
                def wrong(x: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
                    """Perturb token order in [batch, sequence, heads, channels] tensors."""
                    return original(x, group).flip(1)
                patch.setattr(fm, "_sequence_to_heads", wrong)
            module = fm.FlashMaskAttention(softmax_scale=.5)
            module.cp_group = dist.group.WORLD
            actual = module(*local)
            expected = ref.detach().index_select(1, indices)
            if wrong_order:
                with pytest.raises(AssertionError):
                    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
                return
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
            actual.backward(upstream.index_select(1, indices))
            for x, full_x in zip(local, full):
                torch.testing.assert_close(x.grad, full_x.grad.index_select(1, indices), atol=2e-6, rtol=1e-5)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("wrong_order", [False, True], ids=["forward_backward", "negative_token_order"])
def test_ulysses_transport_matches_fp32_oracle(tmp_path: Path, wrong_order: bool) -> None:
    mp.spawn(_worker, args=((tmp_path / "rendezvous").as_uri(), wrong_order), nprocs=2, join=True)


def test_missing_optional_dependency(monkeypatch) -> None:
    monkeypatch.setattr(fm, "safe_import", lambda name: (False, None))
    with pytest.raises(ImportError, match="FlashMask"):
        fm.FlashMaskAttention(softmax_scale=.5)


@pytest.mark.parametrize("kind", ["padding", "packed", "window", "thd"])
def test_reject_unsupported_mask_contracts(kind: str) -> None:
    q = torch.zeros(1, 12, 4, 4)
    k = v = torch.zeros(1, 12, 2, 4)
    mask = torch.ones(1, 12) if kind == "padding" else None
    kwargs = {}
    if kind == "packed":
        kwargs["cu_seqlens"] = torch.tensor([0, 6, 12], dtype=torch.int32)
    if kind == "window":
        kwargs["window_size"] = (4, 0)
    if kind == "thd":
        q, k, v = (x.squeeze(0) for x in (q, k, v))
    with pytest.raises(ValueError):
        preprocess_args_and_kwargs_for_attn(q, k, v, mask, "flashmask", **kwargs)


def test_factory_preserves_bshd_and_column_mask(monkeypatch) -> None:
    monkeypatch.setattr(fm, "safe_import", lambda name: (True, SimpleNamespace(flashmask_attention=_math_attention)))
    module, fn = initialize_attn_module_and_func("flashmask", 4, 4, 4, .5, num_gqa_groups=2)
    q = torch.randn(2, 12, 4, 4, requires_grad=True)
    k = torch.randn(2, 12, 2, 4, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    output = fn(q, k, v)
    expected = _math_attention(q, k, v, causal=True, softmax_scale=.5)
    torch.testing.assert_close(output, expected)
    mask = torch.full((2, 1, 12, 1), 12, dtype=torch.int32)
    q1, k1, v1, kw = preprocess_args_and_kwargs_for_attn(q, k, v, None, "flashmask", startend_row_indices=mask)
    assert q1 is q and k1 is k and v1 is v
    assert kw["startend_row_indices"] is mask
    assert isinstance(module, fm.FlashMaskAttention)
