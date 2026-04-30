"""flash_kmeans_cuda — hand-rolled CUDA kernels for batched Euclidean K-Means.

Public API mirrors ``flash_kmeans.batch_kmeans_Euclid`` so callers can swap
backends with a one-line import change. Cosine, Dot, and large-N streaming
paths are intentionally not provided — use ``flash_kmeans`` for those.
"""

from ._version import __version__
from .kmeans import batch_kmeans_Euclid
from .ops import euclid_assign, centroid_update_sorted, centroid_finalize

__all__ = [
    "__version__",
    "batch_kmeans_Euclid",
    "euclid_assign",
    "centroid_update_sorted",
    "centroid_finalize",
]
