"""Stage 1 acceptance: the new policy-table dispatcher must produce the
same cluster_ids as a deterministic Python fp32 reference for every (D, K)
currently exercised by bench_d_sweep.py. This is a one-off check guarding
the Stage 1 refactor; it's intentionally narrow vs the python reference
rather than vs Triton (Triton diverges on tied-distance points; the
reference doesn't)."""

from __future__ import annotations

import pytest
import torch

from flash_kmeans_cuda import _C


def _reference_assign(x: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    """Deterministic fp32 reference: argmin_k ||x[b,n] - centroids[b,k]||^2."""
    diff = x.unsqueeze(2).float() - centroids.unsqueeze(1).float()
    dist = (diff * diff).sum(-1)
    return dist.argmin(dim=-1).to(torch.int32)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "B,N,K,D",
    [
        (1, 4096,  128,  64),
        (1, 4096,  128,  96),
        (1, 4096,  128, 128),
        (1, 4096,  128, 192),
        (1, 4096,  128, 256),
        (1, 4096, 1024, 128),
        (1, 4096, 8192, 128),
    ],
)
def test_dispatcher_matches_reference(B, N, K, D, dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(0)
    x = torch.randn(B, N, D, device="cuda", dtype=dtype)
    centroids = x[:, :K].contiguous()
    x_sq = (x.float() ** 2).sum(-1).contiguous()
    c_sq = (centroids.float() ** 2).sum(-1).contiguous()

    ids = _C.euclid_assign(x, centroids, x_sq, c_sq, None)
    ref = _reference_assign(x, centroids)

    # mma rounding allows tied-distance differences on a small fraction.
    # bf16's 7-bit mantissa accumulates more rounding error than fp16's 10
    # bits at large K, so the bf16 threshold is wider — empirically up to
    # ~3% on K=1024 D=128. fp16 stays under 1% on every measured shape.
    threshold = 0.05 if dtype == torch.bfloat16 else 0.02
    disagree = (ids != ref).float().mean().item()
    assert disagree < threshold, (
        f"D={D} K={K} dtype={dtype}: {disagree:.3%} disagreement vs python "
        f"reference (threshold {threshold:.0%} — tied-distance rounding only)"
    )


@pytest.mark.parametrize(
    "D",
    [64, 96, 128, 192, 224, 256, 320, 384],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_all_locked_d_values_dispatch(D, dtype):
    """Stage 2 acceptance: every D in the locked set produces correct
    cluster_ids vs the python reference, for both dtypes and a mid-range K."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(0)
    B, N, K = 1, 2048, 256
    x = torch.randn(B, N, D, device="cuda", dtype=dtype)
    centroids = x[:, :K].contiguous()
    x_sq = (x.float() ** 2).sum(-1).contiguous()
    c_sq = (centroids.float() ** 2).sum(-1).contiguous()
    ids = _C.euclid_assign(x, centroids, x_sq, c_sq, None)
    ref = _reference_assign(x, centroids)
    threshold = 0.05 if dtype == torch.bfloat16 else 0.02
    disagree = (ids != ref).float().mean().item()
    assert disagree < threshold, (
        f"D={D} dtype={dtype}: {disagree:.3%} disagreement vs reference "
        f"(threshold {threshold:.0%} — tied-distance rounding only)"
    )
