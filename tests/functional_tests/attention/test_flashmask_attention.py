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

"""Real FlashMask kernels versus an independent FP32 softmax reference."""

import pytest
import torch

from nemo_automodel.components.attention.utils import initialize_attn_module_and_func, preprocess_args_and_kwargs_for_attn


@pytest.mark.parametrize("mask_kind", ["causal", "documents", "empty"])
def test_flashmask_kernel_output_and_gradients(mask_kind: str) -> None:
    pytest.importorskip("flashmask.frontend.torch")
    if not torch.cuda.is_available():
        pytest.skip("FlashMask kernels require a Hopper or Blackwell GPU")
    torch.manual_seed(4397)
    size = 128
    q, k, v = [torch.randn(1, size, h, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True) for h in (4, 2, 2)]
    refs = [t.detach().float().requires_grad_() for t in (q, k, v)]
    positions = torch.arange(size, device="cuda")
    allowed = positions[:, None] >= positions[None, :]
    column_mask = None
    if mask_kind == "documents":
        documents = positions >= 47
        allowed = allowed & (documents[:, None] == documents[None, :])
        column_mask = torch.where(positions < 47, 47, size).to(torch.int32).reshape(1, 1, size, 1)
    elif mask_kind == "empty":
        allowed = torch.zeros_like(allowed)
        column_mask = torch.zeros(1, 1, size, 1, device="cuda", dtype=torch.int32)
    qr = refs[0].transpose(1, 2)
    kr, vr = [t.transpose(1, 2).repeat_interleave(2, dim=1) for t in refs[1:]]
    scores = qr @ kr.transpose(-1, -2) / 128**.5
    scores = scores.masked_fill(~allowed, -torch.inf)
    # Empty rows have zero attention and zero gradients, defined explicitly.
    safe_scores = torch.where(allowed.any(-1)[None, None, :, None], scores, torch.zeros_like(scores))
    probabilities = safe_scores.softmax(-1) * allowed[None, None]
    expected = (probabilities @ vr).transpose(1, 2)
    module, attention = initialize_attn_module_and_func("flashmask", 4, 128, 128, 128**-.5, num_gqa_groups=2)
    q1, k1, v1, kwargs = preprocess_args_and_kwargs_for_attn(q, k, v, None, "flashmask", startend_row_indices=column_mask)
    actual = attention(q1, k1, v1, **kwargs)
    upstream = torch.randn_like(actual)
    actual.backward(upstream)
    expected.backward(upstream.float())
    for tested, reference in [(actual, expected), *[(t.grad, r.grad) for t, r in zip((q, k, v), refs)]]:
        assert torch.isfinite(tested).all()
        difference = tested.float() - reference
        rms = difference.square().mean().sqrt() / reference.square().mean().sqrt().clamp_min(1e-9)
        maximum = difference.abs().max() / reference.abs().max().clamp_min(1e-9)
        assert rms < .01
        assert maximum < .02
