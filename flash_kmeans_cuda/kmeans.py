"""Iteration loop for batched Euclidean K-Means.

Mirrors the structure of
``third_party/flash-kmeans/flash_kmeans/kmeans_triton_impl.py:55-110``: each
iteration runs assign + sorted centroid update + finalize, then computes the
shift to check convergence.
"""

from __future__ import annotations

from typing import Optional, Tuple
import torch

from .ops import euclid_assign, centroid_update_sorted, centroid_finalize


def _euclid_iter(
    x: torch.Tensor,
    x_sq: torch.Tensor,
    centroids: torch.Tensor,
    *,
    compute_shift: bool = True,
    cluster_ids_buf: Optional[torch.Tensor] = None,
    sums_buf: Optional[torch.Tensor] = None,
    counts_buf: Optional[torch.Tensor] = None,
    new_centroids_buf: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    """One assign+update+finalize step. Returns (new_centroids, shift, cluster_ids).

    When ``compute_shift=False`` the shift is None — saves a fp32 cast +
    norm + reduction per iter.

    The optional ``*_buf`` args let the caller reuse pre-allocated buffers
    across iterations to avoid per-iter alloc churn through torch's caching
    allocator.
    """
    B, K, D = centroids.shape
    c_sq = (centroids.float() ** 2).sum(dim=-1).contiguous()
    cluster_ids = euclid_assign(x, centroids, x_sq, c_sq=c_sq, out=cluster_ids_buf)

    sums, counts = centroid_update_sorted(
        x, cluster_ids, K, sums_out=sums_buf, counts_out=counts_buf)
    new_centroids = centroid_finalize(
        sums, counts, centroids, out=new_centroids_buf)

    shift = None
    if compute_shift:
        shift = (new_centroids.float() - centroids.float()).norm(dim=-1).max()
    return new_centroids, shift, cluster_ids


def batch_kmeans_Euclid(
    x: torch.Tensor,
    n_clusters: int,
    max_iters: int = 100,
    tol: float = 0.0,
    init_centroids: Optional[torch.Tensor] = None,
    verbose: bool = False,
    *,
    use_heuristic: bool = True,  # accepted for API parity, ignored
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Batched K-Means clustering using Euclidean distance.

    Drop-in replacement for ``flash_kmeans.batch_kmeans_Euclid``.

    Args:
        x: (B, N, D) compute dtype, CUDA, contiguous.
        n_clusters: K.
        max_iters: maximum number of iterations.
        tol: convergence tolerance on the max centroid shift; <=0 means run all
            ``max_iters`` (matching upstream).
        init_centroids: optional (B, K, D) initial centroids. If None, K random
            points from x are used per batch.
        verbose: print per-iter shift.
        use_heuristic: ignored — the CUDA kernels don't autotune.

    Returns:
        (cluster_ids, centroids, n_iters_run)
        cluster_ids: (B, N) int32
        centroids:   (B, K, D) compute dtype
        n_iters_run: int
    """
    del use_heuristic  # parity with upstream signature

    assert x.is_cuda, "flash_kmeans_cuda requires a CUDA input"
    assert x.dim() == 3, "x must be (B, N, D)"

    B, N, D = x.shape

    x_sq = (x.float() ** 2).sum(dim=-1).contiguous()  # (B, N) fp32

    if init_centroids is None:
        idx = torch.randint(0, N, (B, n_clusters), device=x.device)
        centroids = torch.gather(
            x, dim=1, index=idx.unsqueeze(-1).expand(-1, -1, D)
        ).contiguous()
    else:
        centroids = init_centroids.contiguous()
    centroids = centroids.view(B, n_clusters, D)

    # Skip shift compute when tol<=0 and not verbose: tol=0 means "always run
    # to max_iters" so the shift result is unused. Eliminates a (B,K,D) fp32
    # cast + norm + max + .item() sync per iter — significant at large K.
    need_shift = tol > 0 or verbose

    # Pre-allocate buffers and ping-pong between two centroid buffers to
    # avoid per-iter alloc churn through torch's caching allocator.
    cluster_ids = torch.empty((B, N), device=x.device, dtype=torch.int32)
    sums_buf = torch.zeros((B, n_clusters, D), device=x.device, dtype=torch.float32)
    counts_buf = torch.zeros((B, n_clusters), device=x.device, dtype=torch.int32)
    centroids_b = torch.empty_like(centroids)

    n_iters_run = 0
    cur, nxt = centroids, centroids_b
    for it in range(max_iters):
        new_centroids, shift, cluster_ids = _euclid_iter(
            x, x_sq, cur,
            compute_shift=need_shift,
            cluster_ids_buf=cluster_ids,
            sums_buf=sums_buf,
            counts_buf=counts_buf,
            new_centroids_buf=nxt,
        )
        n_iters_run = it + 1
        if verbose:
            print(f"Iter {it}, center shift: {shift.item():.6f}")
        if need_shift and shift.item() < tol:
            cur = new_centroids
            break
        cur, nxt = nxt, cur  # swap

    return cluster_ids, cur, n_iters_run
