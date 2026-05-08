"""Large-N compatibility API.

The upstream ``flash_kmeans.kmeans_largeN`` streams CPU data to CUDA in chunks
and can distribute work across multiple GPUs. This module provides the same
call signatures with a single-device implementation backed by this package's
Euclidean assignment kernel.
"""

from __future__ import annotations

from typing import Optional, Tuple
import warnings

import torch

from .ops import euclid_assign


def _resolve_device(device: Optional[torch.device]) -> torch.device:
    if device is None:
        if not torch.cuda.is_available():
            raise RuntimeError("kmeans_largeN requires a CUDA device")
        if torch.cuda.device_count() > 1:
            warnings.warn(
                "flash_kmeans_cuda.kmeans_largeN currently uses one CUDA device; "
                "pass device=... to choose it explicitly.",
                RuntimeWarning,
                stacklevel=2,
            )
        return torch.device("cuda:0")
    dev = torch.device(device)
    if dev.type != "cuda":
        raise RuntimeError("kmeans_largeN requires a CUDA device")
    return dev


def _copy_block(x: torch.Tensor, start: int, end: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    non_blocking = x.device.type == "cpu" and x.is_pinned()
    return x[start:end].to(device=device, dtype=dtype, non_blocking=non_blocking).contiguous()


def _init_centroids(
    x: torch.Tensor,
    n_clusters: int,
    device: torch.device,
    dtype: torch.dtype,
    init_centroids: Optional[torch.Tensor],
) -> torch.Tensor:
    if init_centroids is not None:
        return init_centroids.to(device=device, dtype=dtype, non_blocking=True).contiguous()
    N = x.shape[0]
    idx = torch.randint(0, N, (n_clusters,), device="cpu")
    non_blocking = x.device.type == "cpu" and x.is_pinned()
    return x[idx].to(device=device, dtype=dtype, non_blocking=non_blocking).contiguous()


def _assign_block(x_block: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    x_b = x_block.unsqueeze(0)
    c_b = centroids.unsqueeze(0)
    x_sq = (x_b.float() ** 2).sum(dim=-1).contiguous()
    c_sq = (c_b.float() ** 2).sum(dim=-1).contiguous()
    return euclid_assign(x_b, c_b, x_sq, c_sq=c_sq).squeeze(0)


def kmeans_largeN_assign(
    x: torch.Tensor,
    centroids: torch.Tensor,
    BLOCK_N: int = 1048576,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Assign large ``(N, D)`` data to fixed centroids in chunks.

    Returns an ``int32`` tensor on the selected CUDA device.
    """
    if x.dim() != 2 or centroids.dim() != 2:
        raise ValueError("x and centroids must be rank-2 tensors")
    dev = _resolve_device(device)
    work_dtype = dtype or centroids.dtype or x.dtype
    N = x.shape[0]
    centroids_g = centroids.to(device=dev, dtype=work_dtype, non_blocking=True).contiguous()
    cluster_ids = torch.empty((N,), device=dev, dtype=torch.int32)

    for start in range(0, N, BLOCK_N):
        end = min(start + BLOCK_N, N)
        x_block = _copy_block(x, start, end, dev, work_dtype)
        cluster_ids[start:end] = _assign_block(x_block, centroids_g)

    return cluster_ids


def kmeans_largeN(
    x: torch.Tensor,
    n_clusters: int,
    max_iters: int = 100,
    tol: float = 0.0,
    verbose: bool = False,
    BLOCK_N: int = 1048576,
    init_centroids: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Chunked Euclidean K-Means for large ``(N, D)`` inputs.

    This compatibility path uses one CUDA device and returns ``(cluster_ids,
    centroids)`` on that device, matching upstream's single-GPU return shape.
    """
    if x.dim() != 2:
        raise ValueError("x must be (N, D)")
    dev = _resolve_device(device)
    work_dtype = dtype or x.dtype
    N, D = x.shape
    K = int(n_clusters)
    centroids = _init_centroids(x, K, dev, work_dtype, init_centroids)
    cluster_ids = torch.empty((N,), device=dev, dtype=torch.int32)

    n_iters_run = 0
    for it in range(max_iters):
        sums = torch.zeros((K, D), device=dev, dtype=torch.float32)
        counts = torch.zeros((K,), device=dev, dtype=torch.float32)

        for start in range(0, N, BLOCK_N):
            end = min(start + BLOCK_N, N)
            x_block = _copy_block(x, start, end, dev, work_dtype)
            ids = _assign_block(x_block, centroids)
            cluster_ids[start:end] = ids
            ids_long = ids.to(torch.int64)
            sums.index_add_(0, ids_long, x_block.float())
            counts.index_add_(
                0,
                ids_long,
                torch.ones((end - start,), device=dev, dtype=torch.float32),
            )

        means = sums / counts.unsqueeze(-1).clamp_min(1.0)
        empty = counts.eq(0).unsqueeze(-1)
        new_centroids = torch.where(empty, centroids.float(), means).to(work_dtype)
        shift = (new_centroids.float() - centroids.float()).norm(dim=-1).max()
        n_iters_run = it + 1
        if verbose:
            print(f"Iter {it}, center shift: {shift.item():.6f}")
        centroids = new_centroids
        if shift < tol:
            break

    del n_iters_run  # Upstream large-N API returns only labels and centroids.
    return cluster_ids, centroids
