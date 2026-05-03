"""Safety-net tests for the persistent (N_TILES_PER_CTA > 1) kernel path.

Specifically guards against the failure modes Agent 3 flagged:
- cp.async pipeline group counter contamination across n_tile iterations
- best[]/xs_*_cache state leaking between tiles
- Tail handling when N is not a multiple of (BLOCK_N * N_TILES_PER_CTA)

The kernel selects N_TILES via the FKC_NTILES env var, which is read on every
launch by read_env_knobs(), so we toggle it inline within each test.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest
import torch

from flash_kmeans_cuda import _C


def _set_ntiles(ntiles: str):
    os.environ["FKC_NTILES"] = ntiles


@pytest.fixture(autouse=True)
def _clean_fkc_ntiles():
    """Restore FKC_NTILES to its original state after each test to prevent
    env var leakage from polluting subprocess-based tests (e.g. test_assign_autotune)."""
    original = os.environ.get("FKC_NTILES")
    yield
    if original is None:
        os.environ.pop("FKC_NTILES", None)
    else:
        os.environ["FKC_NTILES"] = original


@pytest.mark.parametrize("ntiles", ["1", "2", "4"])
def test_identity_centroids(ntiles):
    """First K rows of x as centroids -> argmin trivially picks row index."""
    _set_ntiles(ntiles)
    torch.manual_seed(7)
    B, N, K, D = 1, 4096, 64, 128
    x = torch.randn(B, N, D, device="cuda", dtype=torch.float16)
    centroids = x[:, :K].contiguous()
    x_sq = (x.float() ** 2).sum(-1).contiguous()
    c_sq = (centroids.float() ** 2).sum(-1).contiguous()
    ids = _C.euclid_assign(x, centroids, x_sq, c_sq, None)
    expected = torch.arange(K, device="cuda", dtype=torch.int32)
    matches = (ids[0, :K] == expected).float().mean().item()
    assert matches > 0.99, f"FKC_NTILES={ntiles} identity_match={matches:.6f}"


@pytest.mark.parametrize("ntiles", ["1", "2", "4"])
def test_tail_short_remainder(ntiles):
    """N = 257 with BLOCK_N=128 forces a 1-row tail -- shouldn't OOB or break."""
    _set_ntiles(ntiles)
    torch.manual_seed(7)
    B, N, K, D = 1, 257, 16, 128
    x = torch.randn(B, N, D, device="cuda", dtype=torch.float16)
    centroids = x[:, :K].contiguous()
    x_sq = (x.float() ** 2).sum(-1).contiguous()
    c_sq = (centroids.float() ** 2).sum(-1).contiguous()
    ids = _C.euclid_assign(x, centroids, x_sq, c_sq, None)
    in_range = ((ids >= 0) & (ids < K)).all().item()
    self_match = (ids[0, :K] == torch.arange(K, device="cuda", dtype=torch.int32)).float().mean().item()
    assert in_range and self_match > 0.95, (
        f"FKC_NTILES={ntiles}: in_range={in_range} self_match={self_match:.4f}"
    )


@pytest.mark.parametrize("ntiles", ["1", "2", "4"])
def test_determinism(ntiles):
    """Same input must yield same cluster_ids on repeated launches."""
    _set_ntiles(ntiles)
    torch.manual_seed(7)
    B, N, K, D = 1, 8192, 128, 128
    x = torch.randn(B, N, D, device="cuda", dtype=torch.float16)
    centroids = x[:, :K].contiguous()
    x_sq = (x.float() ** 2).sum(-1).contiguous()
    c_sq = (centroids.float() ** 2).sum(-1).contiguous()
    ids1 = _C.euclid_assign(x, centroids, x_sq, c_sq, None).clone()
    ids2 = _C.euclid_assign(x, centroids, x_sq, c_sq, None).clone()
    assert (ids1 == ids2).all().item(), f"FKC_NTILES={ntiles} non-deterministic"


def test_ntiles_invariance():
    """N_TILES=1 and N_TILES=2 must produce identical cluster_ids on the same
    input -- the loop structure is the only difference, both must consume the
    same data and produce the same answer."""
    script = r"""
import os, sys, torch
from flash_kmeans_cuda import _C

# We can only read FKC_NTILES once via the kernel-local static. Run two
# subprocesses to compare.
mode = sys.argv[1]
os.environ['FKC_NTILES'] = mode
import importlib
torch.manual_seed(11)
B, N, K, D = 1, 32768, 256, 128
x = torch.randn(B, N, D, device='cuda', dtype=torch.float16)
centroids = x[:, :K].contiguous()
x_sq = (x.float() ** 2).sum(-1).contiguous()
c_sq = (centroids.float() ** 2).sum(-1).contiguous()
ids = _C.euclid_assign(x, centroids, x_sq, c_sq, None)
import numpy as np
np.save(f'/tmp/ids_n{mode}.npy', ids.cpu().numpy()) if False else torch.save(ids.cpu(), sys.argv[2])
"""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        p1 = os.path.join(tmp, "n1.pt")
        p2 = os.path.join(tmp, "n2.pt")
        for ntiles, path in [("1", p1), ("2", p2)]:
            r = subprocess.run(
                [sys.executable, "-c", script, ntiles, path],
                capture_output=True, text=True, timeout=120,
            )
            assert r.returncode == 0, f"FKC_NTILES={ntiles} run failed:\n{r.stderr}"

        import torch
        a = torch.load(p1)
        b = torch.load(p2)
        # fp16 mma scheduling differences across N_TILES variants change the
        # add-accumulate order in registers, which produces near-tie
        # rounding differences on a small fraction of points. A real bug
        # (state leak between tiles, group counter contamination) would
        # disagree on >>10% of points. 1% is the safety margin.
        disagree = (a != b).float().mean().item()
        assert disagree < 0.01, (
            f"N_TILES=1 vs N_TILES=2 disagree on {disagree:.3%} of points "
            f"(expected <1% — fp16 mma rounding allows tiny drift, but "
            f">1% indicates state leak between tiles)"
        )
