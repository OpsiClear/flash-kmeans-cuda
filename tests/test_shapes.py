"""Edge-case shape coverage: tail blocks (N % BLOCK_N != 0), small K, varied D."""

from __future__ import annotations

import pytest
import torch

from flash_kmeans_cuda import batch_kmeans_Euclid


@pytest.mark.parametrize(
    "B,N,K,D",
    [
        (1, 100, 8, 64),         # N tiny, not multiple of BLOCK_N
        (1, 257, 16, 128),       # tail block of size 1
        (1, 1024, 1, 64),        # K=1 edge case
        (1, 1024, 63, 64),       # K = BLOCK_K - 1 (tail K-chunk)
        (1, 1024, 65, 64),       # K = BLOCK_K + 1 (forces 2 K-chunks)
        (1, 1024, 32, 256),      # D = 256
        (3, 2048, 32, 64),       # B > 1
    ],
)
def test_shape(B, N, K, D):
    x = torch.randn(B, N, D, device="cuda", dtype=torch.float16)
    ids, cents, n_iters = batch_kmeans_Euclid(x, K, max_iters=3, tol=0.0)
    assert ids.shape == (B, N)
    assert ids.dtype == torch.int32
    assert cents.shape == (B, K, D)
    assert cents.dtype == torch.float16
    # Each cluster_id must be in [0, K).
    assert (ids >= 0).all() and (ids < K).all()


def test_large_k_raw_kmeans_assignment_matches_reference():
    """K>=8192 D=128 uses the raw assign path where x_sq is intentionally unused."""
    torch.manual_seed(123)
    B, N, K, D = 1, 256, 8192, 128
    x = torch.randn(B, N, D, device="cuda", dtype=torch.float16)
    init = torch.randn(B, K, D, device="cuda", dtype=torch.float16)

    ids, cents, n_iters = batch_kmeans_Euclid(
        x, K, max_iters=1, tol=0.0, init_centroids=init)

    x_sq = (x.float() ** 2).sum(dim=-1).contiguous()
    c_sq = (init.float() ** 2).sum(dim=-1).contiguous()
    cross = torch.einsum("bnd,bkd->bnk", x.float(), init.float())
    ref = (x_sq.unsqueeze(-1) + c_sq.unsqueeze(1) - 2.0 * cross).argmin(dim=-1)
    disagreement = (ids.long() != ref.long()).float().mean().item()

    assert n_iters == 1
    assert cents.shape == (B, K, D)
    assert disagreement < 0.05
