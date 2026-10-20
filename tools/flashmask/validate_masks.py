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

"""Independently check benchmark mask encodings and SDPA references on a GPU."""

import json

import torch
from masks import column_mask, reference


def main() -> None:
    """Check exact masks, FP32 mathematical outputs/gradients and a wrong-mask control."""
    torch.manual_seed(4397)
    seq = 256
    report = {}
    for documents, doc_lengths in [(4, [32, 32, 64, 128]), (16, [8, 8, 16, 32] * 4)]:
        report[str(documents)] = check_profile(seq, documents, doc_lengths)
    print(json.dumps(report, indent=2))


def check_profile(
    seq: int, documents: int, doc_lengths: list[int]
) -> dict[str, dict[str, dict[str, float] | float | int]]:
    """Check a literal length profile independently of the production boundary helper."""
    doc_ids = torch.repeat_interleave(torch.arange(documents, device="cuda"), torch.tensor(doc_lengths, device="cuda"))
    literal_starts = [sum(doc_lengths[:i]) for i in range(documents)]
    starts = torch.tensor(literal_starts, device="cuda")[doc_ids]
    ends = starts + torch.tensor(doc_lengths, device="cuda")[doc_ids]
    pos = torch.arange(seq, device="cuda")
    full = [torch.randn(1, seq, h, 128, device="cuda", dtype=torch.bfloat16) for h in (32, 4, 4)]
    report = {}
    for kind in ("documents", "prefix"):
        allowed = (doc_ids[:, None] == doc_ids[None, :]) & (pos[:, None] >= pos[None, :])
        if kind == "prefix":
            allowed |= (doc_ids[:, None] == doc_ids[None, :]) & (pos[None, :] < ((starts + ends) // 2)[None, :])
        encoded = column_mask(seq, kind, documents)[0, 0]
        decoded = (pos[:, None] < encoded[None, :, 0]) & (pos[:, None] >= encoded[None, :, 1])
        assert torch.equal(allowed, decoded), kind
        oracle_inputs = [t.float().requires_grad_() for t in full]
        q, k, v = (t.transpose(1, 2) for t in oracle_inputs)
        k, v = (t.repeat_interleave(8, dim=1) for t in (k, v))
        scores = q @ k.transpose(-1, -2) / 128**0.5
        expected = (scores.masked_fill(~allowed, -torch.inf).softmax(-1) @ v).transpose(1, 2)
        wrong = (scores.masked_fill(~(pos[:, None] >= pos[None, :]), -torch.inf).softmax(-1) @ v).transpose(1, 2)
        bad_rms = (wrong - expected).square().mean().sqrt() / expected.square().mean().sqrt()
        assert bad_rms.item() > 0.2
        actual_inputs = [t.detach().requires_grad_() for t in full]
        actual = reference(*actual_inputs, kind, documents)
        grad = torch.randn_like(actual)
        expected.backward(grad.float())
        actual.backward(grad)
        checks = {}
        for name, a, b in [("output", actual, expected)] + [
            (f"grad_{i}", a.grad, b.grad) for i, (a, b) in enumerate(zip(actual_inputs, oracle_inputs))
        ]:
            rms = ((a.float() - b.float()).square().mean().sqrt() / b.float().square().mean().sqrt()).item()
            assert rms < 0.01, (kind, name, rms)
            checks[name] = rms
        report[kind] = {
            "relative_rms": checks,
            "wrong_global_causal_relative_rms": bad_rms.item(),
            "mask_elements": allowed.sum().item(),
        }
    return report


if __name__ == "__main__":
    main()
