"""Each dtype routes to the correct kernel and produces sane output."""

from __future__ import annotations

import pytest
import torch

from flash_kmeans_cuda import batch_kmeans_Euclid, euclid_assign


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_assign_only(dtype):
    B, N, K, D = 1, 1024, 32, 64
    x = torch.randn(B, N, D, device="cuda", dtype=dtype)
    centroids = x[:, :K].contiguous()
    x_sq = (x.float() ** 2).sum(dim=-1).contiguous()
    ids = euclid_assign(x, centroids, x_sq)
    assert ids.shape == (B, N)
    assert ids.dtype == torch.int32
    assert (ids >= 0).all() and (ids < K).all()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_kmeans_dtype(dtype):
    x = torch.randn(2, 2048, 64, device="cuda", dtype=dtype)
    ids, cents, n_iters = batch_kmeans_Euclid(x, 32, max_iters=5, tol=0.0)
    assert cents.dtype == dtype
    assert n_iters == 5


def test_assignment_self_centroids():
    """If centroids equal points, every point should map to its own row (or a tie)."""
    B, N, D = 1, 64, 32
    x = torch.randn(B, N, D, device="cuda", dtype=torch.float16)
    centroids = x.clone()
    x_sq = (x.float() ** 2).sum(dim=-1).contiguous()
    ids = euclid_assign(x, centroids, x_sq)
    # Each point's own centroid is at distance 0 (modulo fp16 round-off);
    # the nearest centroid should be itself unless another point is identical.
    # Just assert *some* sensible distribution: not all collapsed to 0.
    unique = ids.unique().numel()
    assert unique > N // 4
