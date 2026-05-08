"""flash_kmeans_cuda — hand-rolled CUDA kernels for batched Euclidean K-Means.

Public API mirrors the common ``flash_kmeans`` entry points. Euclidean CUDA
workloads use the custom kernels; cosine and dot-product entry points use CUDA
similarity assignment on CUDA tensors, and large-N entry points stream chunks
through the Euclidean CUDA backend.
"""

from ._version import __version__
from .interface import FlashKMeans
from .kmeans import batch_kmeans_Euclid
from .large import kmeans_largeN, kmeans_largeN_assign
from .ops import euclid_assign, similarity_assign, centroid_update_sorted, centroid_finalize
from .torch_fallback import (
    batch_kmeans_Cosine,
    batch_kmeans_Dot,
    batch_kmeans_Euclid_torch_native,
    triton_centroid_update_euclid,
    triton_centroid_update_sorted_euclid,
)

__all__ = [
    "__version__",
    "FlashKMeans",
    "batch_kmeans_Euclid",
    "batch_kmeans_Cosine",
    "batch_kmeans_Dot",
    "batch_kmeans_Euclid_torch_native",
    "euclid_assign",
    "similarity_assign",
    "centroid_update_sorted",
    "centroid_finalize",
    "triton_centroid_update_euclid",
    "triton_centroid_update_sorted_euclid",
    "kmeans_largeN",
    "kmeans_largeN_assign",
]
