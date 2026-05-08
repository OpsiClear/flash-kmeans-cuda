"""flash-kmeans compatible object-oriented API."""

from __future__ import annotations

from typing import Optional
import warnings

import torch

from .kmeans import batch_kmeans_Euclid
from .large import kmeans_largeN, kmeans_largeN_assign
from .ops import euclid_assign
from .torch_fallback import batch_kmeans_Euclid_torch_native, euclid_assign_torch_native_chunked


class FlashKMeans:
    """Faiss/sklearn-style wrapper compatible with ``flash_kmeans.FlashKMeans``.

    Euclidean CUDA workloads use this package's hand-written kernels. The
    ``use_triton`` argument is kept for API parity; in this package it means
    "use the optimized CUDA backend when possible".
    """

    def __init__(
        self,
        d: int,
        k: int,
        niter: int = 25,
        tol: float = 1e-8,
        use_triton: bool = True,
        seed: int = 0,
        chunk_size_data: int = 32768,
        chunk_size_centroids: int = 1024,
        chunk_size_data_cpu: int = 1048576,
        verbose: bool = False,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        self.d = int(d)
        self.k = int(k)
        self.niter = int(niter)
        self.tol = float(tol)
        self.use_triton = bool(use_triton)
        self.seed = int(seed)
        self.chunk_size_data = int(chunk_size_data)
        self.chunk_size_centroids = int(chunk_size_centroids)
        self.chunk_size_data_cpu = int(chunk_size_data_cpu)
        self.verbose = bool(verbose)
        self.dtype = dtype
        self._raw_device = device

        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        if self.use_triton and self.device.type != "cuda":
            warnings.warn(
                "Optimized flash_kmeans_cuda backend requires CUDA; using torch fallback.",
                RuntimeWarning,
                stacklevel=2,
            )
            self.use_triton = False

        self.centroids_b: Optional[torch.Tensor] = None
        self.cluster_ids_b: Optional[torch.Tensor] = None
        self._batch_size: Optional[int] = None

    def _normalize_input(self, data: torch.Tensor) -> tuple[torch.Tensor, Optional[int], int, int]:
        if data.ndim == 2:
            N, D = data.shape
            return data.unsqueeze(0), None, N, D
        if data.ndim == 3:
            B, N, D = data.shape
            return data, B, N, D
        raise ValueError("data must be of shape (n_samples, n_features) or (batch_size, n_samples, n_features)")

    def train(self, data: torch.Tensor):
        """Fit K-Means on ``data`` and store centroids/labels."""
        x_b, B, N, D = self._normalize_input(data)
        if D != self.d:
            raise ValueError(f"expected feature dimension d={self.d}, got D={D}")

        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        if data.device.type == "cpu" and N > self.chunk_size_data_cpu:
            if B is not None:
                raise AssertionError("Batched data with large N on CPU is not supported.")
            labels, centroids = kmeans_largeN(
                x_b[0],
                self.k,
                max_iters=self.niter,
                tol=self.tol,
                verbose=self.verbose,
                BLOCK_N=self.chunk_size_data_cpu,
                device=self._raw_device,
                dtype=self.dtype,
            )
            self.cluster_ids_b = labels.unsqueeze(0)
            self.centroids_b = centroids.unsqueeze(0)
            self._batch_size = B
            return

        compute_dtype = self.dtype or x_b.dtype
        x_b = x_b.to(device=self.device, dtype=compute_dtype, copy=False).contiguous()

        if self.use_triton:
            labels, centroids, _ = batch_kmeans_Euclid(
                x_b,
                self.k,
                max_iters=self.niter,
                tol=self.tol,
                init_centroids=None,
                verbose=self.verbose,
            )
        else:
            labels, centroids, _ = batch_kmeans_Euclid_torch_native(
                x_b,
                self.k,
                max_iters=self.niter,
                tol=self.tol,
                init_centroids=None,
                verbose=self.verbose,
                chunk_size_N=self.chunk_size_data,
                chunk_size_K=self.chunk_size_centroids,
            )

        self.cluster_ids_b = labels
        self.centroids_b = centroids
        self._batch_size = B

    def fit(self, data: torch.Tensor):
        """Alias for ``train``; returns ``self``."""
        self.train(data)
        return self

    def predict(self, data: torch.Tensor) -> torch.Tensor:
        """Assign points to the fitted centroids."""
        if self.centroids_b is None:
            raise RuntimeError("Model not trained. Call train() or fit() first.")

        x_b, B, N, D = self._normalize_input(data)
        if D != self.d:
            raise ValueError(f"expected feature dimension d={self.d}, got D={D}")
        if B != self._batch_size:
            raise ValueError(
                f"Model was trained with batch size B={self._batch_size}, "
                f"but predict received B={B}. Provide matching batch size."
            )

        if data.device.type == "cpu" and N > self.chunk_size_data_cpu:
            if B is not None:
                raise AssertionError("Batched data with large N on CPU is not supported.")
            labels = kmeans_largeN_assign(
                x_b[0],
                self.centroids_b[0],
                BLOCK_N=self.chunk_size_data_cpu,
                device=self._raw_device,
                dtype=self.dtype,
            )
            return labels

        compute_dtype = self.dtype or x_b.dtype
        x_b = x_b.to(device=self.device, dtype=compute_dtype, copy=False).contiguous()
        centroids = self.centroids_b.to(device=self.device, dtype=compute_dtype, copy=False).contiguous()

        if self.use_triton:
            x_sq = (x_b.float() ** 2).sum(dim=-1).contiguous()
            c_sq = (centroids.float() ** 2).sum(dim=-1).contiguous()
            labels_b = euclid_assign(x_b, centroids, x_sq, c_sq=c_sq)
        else:
            x_sq = (x_b.float() ** 2).sum(dim=-1)
            labels_b = euclid_assign_torch_native_chunked(
                x_b,
                centroids,
                x_sq,
                chunk_size_N=self.chunk_size_data,
                chunk_size_K=self.chunk_size_centroids,
            )

        if B is None:
            return labels_b.squeeze(0)
        return labels_b

    def fit_predict(self, data: torch.Tensor) -> torch.Tensor:
        """Fit K-Means and return labels."""
        self.train(data)
        if self.cluster_ids_b is None:
            raise RuntimeError("training did not produce cluster IDs")
        if self._batch_size is None:
            return self.cluster_ids_b.squeeze(0)
        return self.cluster_ids_b
