"""Torch fallback routines used for flash-kmeans API compatibility.

These functions intentionally favor predictable behavior and bounded memory
for fallback code. CUDA tensors use the optimized assignment/update ops where
available; chunked PyTorch remains the CPU and compatibility fallback.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from .ops import centroid_finalize, centroid_update_sorted, similarity_assign


def _choose_centroids(
    x: torch.Tensor,
    n_clusters: int,
    init_centroids: Optional[torch.Tensor],
) -> torch.Tensor:
    B, N, D = x.shape
    if init_centroids is not None:
        return init_centroids.to(device=x.device, dtype=x.dtype).contiguous().view(B, n_clusters, D)

    indices = torch.randint(0, N, (B, n_clusters), device=x.device)
    return torch.gather(
        x,
        dim=1,
        index=indices.unsqueeze(-1).expand(-1, -1, D),
    ).contiguous()


def euclid_assign_torch_native_chunked(
    x: torch.Tensor,
    centroids: torch.Tensor,
    x_sq: torch.Tensor,
    chunk_size_N: int = 32768,
    chunk_size_K: int = 1024,
) -> torch.Tensor:
    """Chunked PyTorch Euclidean assignment.

    Returns ``int32`` labels with shape ``(B, N)``.
    """
    B, N, D = x.shape
    K = centroids.shape[1]
    if centroids.shape != (B, K, D):
        raise ValueError("centroids must be (B, K, D) matching x")

    c_sq = (centroids.float() ** 2).sum(dim=-1)
    cluster_ids = torch.empty((B, N), dtype=torch.int32, device=x.device)

    for n_start in range(0, N, chunk_size_N):
        n_end = min(n_start + chunk_size_N, N)
        x_chunk = x[:, n_start:n_end, :].float()
        x_sq_chunk = x_sq[:, n_start:n_end].float()
        best_dist = torch.full(
            (B, n_end - n_start),
            float("inf"),
            device=x.device,
            dtype=torch.float32,
        )
        best_idx = torch.zeros((B, n_end - n_start), dtype=torch.int64, device=x.device)

        for k_start in range(0, K, chunk_size_K):
            k_end = min(k_start + chunk_size_K, K)
            c_chunk = centroids[:, k_start:k_end, :].float()
            dist = (
                x_sq_chunk.unsqueeze(-1)
                - 2.0 * torch.bmm(x_chunk, c_chunk.transpose(1, 2))
                + c_sq[:, k_start:k_end].unsqueeze(1)
            )
            local_dist, local_idx = dist.min(dim=-1)
            update = local_dist < best_dist
            best_dist = torch.where(update, local_dist, best_dist)
            best_idx = torch.where(update, local_idx + k_start, best_idx)

        cluster_ids[:, n_start:n_end] = best_idx.to(torch.int32)

    return cluster_ids


def similarity_assign_torch_chunked(
    x: torch.Tensor,
    centroids: torch.Tensor,
    *,
    chunk_size_N: int = 32768,
    chunk_size_K: int = 1024,
) -> torch.Tensor:
    """Assign to the centroid with maximum dot product."""
    B, N, D = x.shape
    K = centroids.shape[1]
    if centroids.shape != (B, K, D):
        raise ValueError("centroids must be (B, K, D) matching x")

    cluster_ids = torch.empty((B, N), dtype=torch.int32, device=x.device)
    for n_start in range(0, N, chunk_size_N):
        n_end = min(n_start + chunk_size_N, N)
        x_chunk = x[:, n_start:n_end, :].float()
        best_score = torch.full(
            (B, n_end - n_start),
            -float("inf"),
            device=x.device,
            dtype=torch.float32,
        )
        best_idx = torch.zeros((B, n_end - n_start), dtype=torch.int64, device=x.device)

        for k_start in range(0, K, chunk_size_K):
            k_end = min(k_start + chunk_size_K, K)
            c_chunk = centroids[:, k_start:k_end, :].float()
            score = torch.bmm(x_chunk, c_chunk.transpose(1, 2))
            local_score, local_idx = score.max(dim=-1)
            update = local_score > best_score
            best_score = torch.where(update, local_score, best_score)
            best_idx = torch.where(update, local_idx + k_start, best_idx)

        cluster_ids[:, n_start:n_end] = best_idx.to(torch.int32)

    return cluster_ids


def similarity_assign_auto(
    x: torch.Tensor,
    centroids: torch.Tensor,
    *,
    chunk_size_N: int = 32768,
    chunk_size_K: int = 1024,
) -> torch.Tensor:
    """Similarity assignment with CUDA fast path and torch fallback."""
    if x.is_cuda and centroids.is_cuda:
        return similarity_assign(x.contiguous(), centroids.contiguous())
    return similarity_assign_torch_chunked(
        x,
        centroids,
        chunk_size_N=chunk_size_N,
        chunk_size_K=chunk_size_K,
    )


def centroid_update_torch(
    x: torch.Tensor,
    cluster_ids: torch.Tensor,
    old_centroids: torch.Tensor,
    *,
    normalize: bool = False,
) -> torch.Tensor:
    """Update centroids with fp32 accumulation and preserve empty clusters."""
    B, N, D = x.shape
    K = old_centroids.shape[1]
    if x.is_cuda and cluster_ids.is_cuda and old_centroids.is_cuda:
        sums, counts = centroid_update_sorted(x, cluster_ids.to(torch.int32), K)
        centroids = centroid_finalize(sums, counts, old_centroids)
        if normalize:
            centroids = F.normalize(centroids, p=2, dim=-1)
        return centroids

    sums = torch.zeros((B, K, D), device=x.device, dtype=torch.float32)
    counts = torch.zeros((B, K), device=x.device, dtype=torch.float32)
    ids_long = cluster_ids.to(torch.int64)

    for b in range(B):
        sums[b].index_add_(0, ids_long[b], x[b].float())
        counts[b].index_add_(0, ids_long[b], torch.ones((N,), device=x.device, dtype=torch.float32))

    means = sums / counts.unsqueeze(-1).clamp_min(1.0)
    empty = counts.eq(0).unsqueeze(-1)
    centroids = torch.where(empty, old_centroids.float(), means).to(x.dtype)
    if normalize:
        centroids = F.normalize(centroids, p=2, dim=-1)
    return centroids


def triton_centroid_update_sorted_euclid(
    x: torch.Tensor,
    cluster_ids: torch.Tensor,
    old_centroids: torch.Tensor,
    *,
    BLOCK_N: int = 256,
    centroid_sums: Optional[torch.Tensor] = None,
    centroid_cnts: Optional[torch.Tensor] = None,
    calculate_new: bool = True,
) -> Optional[torch.Tensor]:
    """Compatibility wrapper for upstream's Triton-named Euclidean update.

    ``BLOCK_N`` is accepted for API parity and is not used by this torch path.
    """
    del BLOCK_N
    B, N, D = x.shape
    K = old_centroids.shape[1]
    if x.is_cuda and cluster_ids.is_cuda and old_centroids.is_cuda:
        centroid_sums, centroid_cnts = centroid_update_sorted(
            x,
            cluster_ids.to(torch.int32),
            K,
            sums_out=centroid_sums,
            counts_out=centroid_cnts,
        )
        if not calculate_new:
            return None
        return centroid_finalize(centroid_sums, centroid_cnts, old_centroids)

    if centroid_sums is None:
        centroid_sums = torch.zeros((B, K, D), device=x.device, dtype=torch.float32)
    else:
        centroid_sums.zero_()
    if centroid_cnts is None:
        centroid_cnts = torch.zeros((B, K), device=x.device, dtype=torch.int32)
    else:
        centroid_cnts.zero_()

    ids_long = cluster_ids.to(torch.int64)
    for b in range(B):
        centroid_sums[b].index_add_(0, ids_long[b], x[b].float())
        centroid_cnts[b].index_add_(
            0,
            ids_long[b],
            torch.ones((N,), device=x.device, dtype=torch.int32),
        )

    if not calculate_new:
        return None

    means = centroid_sums / centroid_cnts.float().unsqueeze(-1).clamp_min(1.0)
    empty = centroid_cnts.eq(0).unsqueeze(-1)
    return torch.where(empty, old_centroids.float(), means).to(x.dtype)


def triton_centroid_update_euclid(
    x: torch.Tensor,
    cluster_ids: torch.Tensor,
    old_centroids: torch.Tensor,
) -> torch.Tensor:
    return triton_centroid_update_sorted_euclid(x, cluster_ids, old_centroids)


def _similarity_kmeans(
    x: torch.Tensor,
    n_clusters: int,
    *,
    max_iters: int,
    tol: float,
    init_centroids: Optional[torch.Tensor],
    verbose: bool,
    normalize_input: bool,
    normalize_initial_centroids: bool,
    normalize_centroids: bool,
    label: str,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    if x.dim() != 3:
        raise ValueError("x must be (B, N, D)")
    x_work = F.normalize(x, p=2, dim=-1) if normalize_input else x
    centroids = _choose_centroids(x_work, n_clusters, init_centroids)
    if normalize_initial_centroids:
        centroids = F.normalize(centroids, p=2, dim=-1)

    cluster_ids = torch.empty((x.shape[0], x.shape[1]), device=x.device, dtype=torch.int32)
    n_iters_run = 0
    for it in range(max_iters):
        cluster_ids = similarity_assign_auto(x_work, centroids)
        new_centroids = centroid_update_torch(
            x_work,
            cluster_ids,
            centroids,
            normalize=normalize_centroids,
        )
        shift = (new_centroids.float() - centroids.float()).norm(dim=-1).max()
        n_iters_run = it + 1
        if verbose:
            prefix = f"Iter {it}" if label == "cosine" else f"Iter {it} ({label})"
            print(f"{prefix}, center shift: {shift.item():.6f}")
        if shift < tol:
            centroids = new_centroids
            break
        centroids = new_centroids.clone()

    return cluster_ids, centroids, n_iters_run


def batch_kmeans_Cosine(
    x: torch.Tensor,
    n_clusters: int,
    max_iters: int = 100,
    tol: float = 0.0,
    init_centroids: Optional[torch.Tensor] = None,
    verbose: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Batched K-Means using cosine similarity, matching upstream API."""
    return _similarity_kmeans(
        x,
        n_clusters,
        max_iters=max_iters,
        tol=tol,
        init_centroids=init_centroids,
        verbose=verbose,
        normalize_input=True,
        normalize_initial_centroids=True,
        normalize_centroids=True,
        label="cosine",
    )


def batch_kmeans_Dot(
    x: torch.Tensor,
    n_clusters: int,
    max_iters: int = 100,
    tol: float = 0.0,
    init_centroids: Optional[torch.Tensor] = None,
    verbose: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Batched K-Means using raw dot-product assignment.

    Upstream updates dot-product centroids through its cosine update path, so
    this compatibility implementation also normalizes centroids after updates.
    """
    return _similarity_kmeans(
        x,
        n_clusters,
        max_iters=max_iters,
        tol=tol,
        init_centroids=init_centroids,
        verbose=verbose,
        normalize_input=False,
        normalize_initial_centroids=False,
        normalize_centroids=True,
        label="dot",
    )


def batch_kmeans_Euclid_torch_native(
    x: torch.Tensor,
    n_clusters: int,
    max_iters: int = 100,
    tol: float = 0.0,
    init_centroids: Optional[torch.Tensor] = None,
    verbose: bool = False,
    chunk_size_N: int = 32768,
    chunk_size_K: int = 1024,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Pure PyTorch Euclidean fallback with upstream-compatible signature."""
    if x.dim() != 3:
        raise ValueError("x must be (B, N, D)")
    B, _, _ = x.shape
    x_sq = (x.float() ** 2).sum(dim=-1)
    centroids = _choose_centroids(x, n_clusters, init_centroids)
    cluster_ids = torch.empty((B, x.shape[1]), device=x.device, dtype=torch.int32)

    n_iters_run = 0
    for it in range(max_iters):
        cluster_ids = euclid_assign_torch_native_chunked(
            x, centroids, x_sq, chunk_size_N=chunk_size_N, chunk_size_K=chunk_size_K)
        new_centroids = centroid_update_torch(x, cluster_ids, centroids)
        shift = (new_centroids.float() - centroids.float()).norm(dim=-1).max()
        n_iters_run = it + 1
        if verbose:
            print(f"Iter {it}, center shift: {shift.item():.6f}")
        if shift < tol:
            centroids = new_centroids
            break
        centroids = new_centroids.clone()

    return cluster_ids, centroids, n_iters_run
