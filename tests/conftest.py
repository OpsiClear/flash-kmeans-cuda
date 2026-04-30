"""pytest fixtures and helpers shared across the test suite."""

from __future__ import annotations

import os
import sys
import tempfile

import pytest
import torch

# Force a local Triton cache to avoid Windows TMP path issues — same pattern as
# third_party/flash-kmeans/examples/benchmark_backends.py:12-19.
_LOCAL_TMP = os.path.abspath(".tmp")
os.makedirs(_LOCAL_TMP, exist_ok=True)
os.environ.setdefault("TMP", _LOCAL_TMP)
os.environ.setdefault("TEMP", _LOCAL_TMP)
os.environ.setdefault("TMPDIR", _LOCAL_TMP)
tempfile.tempdir = _LOCAL_TMP
os.environ.setdefault("TRITON_CACHE_DIR", os.path.abspath(".triton_cache"))
os.makedirs(os.environ["TRITON_CACHE_DIR"], exist_ok=True)


def _require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


@pytest.fixture(autouse=True)
def _cuda_required():
    _require_cuda()


def _try_import_triton_oracle():
    """Import the upstream Triton implementation, or skip if unavailable."""
    try:
        from flash_kmeans import batch_kmeans_Euclid as triton_kmeans
        return triton_kmeans
    except Exception as e:  # pragma: no cover
        pytest.skip(f"flash_kmeans Triton oracle unavailable: {e}")


@pytest.fixture
def triton_kmeans():
    return _try_import_triton_oracle()
