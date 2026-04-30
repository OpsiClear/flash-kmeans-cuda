"""Regression test for the sm_80 mma.sync kernel against a torch reference.

The mma kernel is the default for fp16/bf16 since closing the perf gap.
This test verifies it stays correct when changes are made — independent of
the Triton oracle, since the safe path can be re-enabled via
``FKC_ASSIGN_FORCE_SAFE=1`` when iterating.
"""

from __future__ import annotations

import os

import pytest
import torch


def _arch_at_least_sm80():
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability(0)
    return major >= 8


pytestmark = pytest.mark.skipif(
    not _arch_at_least_sm80(), reason="requires sm_80 or newer"
)


def _torch_reference(x, centroids, x_sq, c_sq):
    cross = torch.einsum("bnd,bkd->bnk", x.float(), centroids.float())
    dist = x_sq.unsqueeze(-1) + c_sq.unsqueeze(1) - 2.0 * cross
    return dist.argmin(dim=-1).to(torch.int32)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "B,N,K,D",
    [
        (1, 1024, 64, 64),
        (1, 4096, 128, 128),
        (2, 8192, 256, 128),
    ],
)
def test_mma_matches_reference(dtype, B, N, K, D):
    """sm_80 mma path must match a torch fp32 einsum reference."""
    from flash_kmeans_cuda import euclid_assign

    g = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(B, N, D, generator=g, device="cuda", dtype=dtype)
    centroids = x[:, :K].contiguous()
    x_sq = (x.float() ** 2).sum(dim=-1).contiguous()
    c_sq = (centroids.float() ** 2).sum(dim=-1).contiguous()

    mma_ids = euclid_assign(x, centroids, x_sq, c_sq=c_sq)
    ref_ids = _torch_reference(x, centroids, x_sq, c_sq)

    disagreements = (mma_ids.long() != ref_ids.long()).sum().item()
    total = mma_ids.numel()
    frac = disagreements / total
    # Tensor-core fp16/bf16 accumulation has reduced precision; near tied
    # points may pick a different cluster than fp32 reference. Allow up to 1%.
    assert frac < 1e-2, (
        f"{disagreements}/{total} ({frac:.2%}) sm_80 vs torch reference disagreements"
    )
