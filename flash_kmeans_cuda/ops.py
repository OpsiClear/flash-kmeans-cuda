"""Thin Python wrappers around the C++/CUDA ops.

Normalizes CUDA tensors to contiguous storage, computes optional squared
norms, and allocates output buffers when callers don't pass them. Shape,
dtype, and device validation is shared with the C++ API in ``csrc/api.cpp``.
"""

from __future__ import annotations

from typing import Optional, Tuple
import torch

try:
    from . import _C
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "flash_kmeans_cuda._C C++/CUDA extension is not available. "
        "Run `uv pip install -e . --no-build-isolation` inside the repository "
        "root to build it."
    ) from e


def _ensure_contig(t: torch.Tensor, name: str) -> torch.Tensor:
    if not t.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor (got {t.device})")
    if not t.is_contiguous():
        return t.contiguous()
    return t


def euclid_assign(
    x: torch.Tensor,
    centroids: torch.Tensor,
    x_sq: torch.Tensor,
    c_sq: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Assign each point to its nearest centroid (Euclidean).

    Args:
        x: (B, N, D) fp16/bf16/fp32, contiguous, CUDA.
        centroids: (B, K, D) same dtype as x.
        x_sq: (B, N) fp32 — pre-computed ||x||^2 per point.
        c_sq: (B, K) fp32 — pre-computed ||c||^2 per centroid; computed inline if None.
        out: optional (B, N) int32 buffer to write into.

    Returns:
        cluster_ids: (B, N) int32.
    """
    x = _ensure_contig(x, "x")
    centroids = _ensure_contig(centroids, "centroids")
    x_sq = _ensure_contig(x_sq, "x_sq").to(torch.float32)
    if c_sq is None:
        c_sq = (centroids.float() ** 2).sum(dim=-1)
    c_sq = _ensure_contig(c_sq, "c_sq").to(torch.float32)
    return _C.euclid_assign(x, centroids, x_sq, c_sq, out)


def similarity_assign(
    x: torch.Tensor,
    centroids: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Assign each point to the centroid with maximum dot product.

    This is the CUDA dot-product argmax assignment path used by cosine/dot
    compatibility APIs. It avoids norm buffers and does not materialize a full
    similarity matrix in PyTorch.
    """
    x = _ensure_contig(x, "x")
    centroids = _ensure_contig(centroids, "centroids")
    return _C.similarity_assign(x, centroids, out)


def centroid_update_sorted(
    x: torch.Tensor,
    cluster_ids: torch.Tensor,
    K: int,
    *,
    sums_out: Optional[torch.Tensor] = None,
    counts_out: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sort cluster_ids along N (host op), gather x rows in the same order,
    then accumulate per-cluster sums/counts.

    Args:
        x: (B, N, D) compute dtype.
        cluster_ids: (B, N) int32 (any order).
        K: number of clusters.
        sums_out: optional pre-zeroed (B, K, D) fp32.
        counts_out: optional pre-zeroed (B, K) int32.

    Returns:
        (centroid_sums, centroid_counts).
    """
    x = _ensure_contig(x, "x")
    cluster_ids = _ensure_contig(cluster_ids, "cluster_ids")
    if cluster_ids.dtype != torch.int32:
        cluster_ids = cluster_ids.to(torch.int32)

    B, N, D = x.shape

    # Sort by cluster_id along N. For very large K, avoiding a full x_sorted
    # materialization wins; below that, contiguous x_sorted reads beat the
    # indexed kernel's random row loads.
    sorted_ids, perm = torch.sort(cluster_ids, dim=1, stable=False)
    sorted_ids = sorted_ids.contiguous()

    if sums_out is None:
        sums_out = torch.zeros(
            (B, K, D), device=x.device, dtype=torch.float32
        )
    else:
        sums_out.zero_()
    if counts_out is None:
        counts_out = torch.zeros(
            (B, K), device=x.device, dtype=torch.int32
        )
    else:
        counts_out.zero_()

    if x.dtype == torch.float16 and D == 128 and K >= 256:
        perm = perm.contiguous().to(torch.int32)
        _C.centroid_update_sorted_indexed(x, perm, sorted_ids, sums_out, counts_out)
    else:
        perm_exp = perm.unsqueeze(-1).expand(-1, -1, D)
        x_sorted = torch.gather(x, dim=1, index=perm_exp).contiguous()
        _C.centroid_update_sorted(x_sorted, sorted_ids, sums_out, counts_out)
    return sums_out, counts_out


def centroid_finalize(
    centroid_sums: torch.Tensor,
    centroid_counts: torch.Tensor,
    old_centroids: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """new[b,k] = sums[b,k] / counts[b,k] if counts > 0 else old_centroids[b,k].

    Casts back to old_centroids.dtype.
    """
    return _C.centroid_finalize(
        centroid_sums.contiguous(),
        centroid_counts.contiguous(),
        old_centroids.contiguous(),
        out,
    )
