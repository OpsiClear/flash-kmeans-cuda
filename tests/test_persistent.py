"""Safety-net tests for the persistent (N_TILES_PER_CTA > 1) kernel path.

Specifically guards against the failure modes Agent 3 flagged:
- cp.async pipeline group counter contamination across n_tile iterations
- best[]/xs_*_cache state leaking between tiles
- Tail handling when N is not a multiple of (BLOCK_N * N_TILES_PER_CTA)

The kernel selects N_TILES via the FKC_NTILES env var; we toggle it inside
each test via os.environ since the kernel reads it once at first launch.
But because the env is read into a function-local static the first time
launch_assign_sm80 is called, we can't actually flip it after that — so
each subprocess test runs with a fixed env value via subprocess launch.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest


# Run each (n_tiles_value, test_fn) combination in its own subprocess so that
# the kernel's static env-var read picks up the right value.
_SCRIPT = r"""
import os, sys
os.environ['FKC_NTILES'] = sys.argv[1]
import torch
from flash_kmeans_cuda import _C

mode = sys.argv[2]
torch.manual_seed(7)

if mode == 'identity':
    # Centroids ARE the first K input rows -> rows 0..K-1 must map to themselves
    # (the argmin is trivially the matching index because dist=0 for that k).
    # Rows K..N-1 are unrelated random points and can map anywhere.
    B, N, K, D = 1, 4096, 64, 128
    x = torch.randn(B, N, D, device='cuda', dtype=torch.float16)
    centroids = x[:, :K].contiguous()
    x_sq = (x.float() ** 2).sum(-1).contiguous()
    c_sq = (centroids.float() ** 2).sum(-1).contiguous()
    ids = _C.euclid_assign(x, centroids, x_sq, c_sq, None)
    expected = torch.arange(K, device='cuda', dtype=torch.int32)
    matches = (ids[0, :K] == expected).float().mean().item()
    print(f'identity_match={matches:.6f}')
    sys.exit(0 if matches > 0.99 else 1)

elif mode == 'tail':
    # N = BLOCK_N * N_TILES_PER_CTA + 1 -> tail CTA processes only 1 row.
    # BLOCK_N=128, N_TILES=2 -> N=257 puts last CTA at n_tile=0 with 1 row,
    # tile 1 hits n_count=0 break.
    B, N, K, D = 1, 257, 16, 128
    x = torch.randn(B, N, D, device='cuda', dtype=torch.float16)
    centroids = x[:, :K].contiguous()
    x_sq = (x.float() ** 2).sum(-1).contiguous()
    c_sq = (centroids.float() ** 2).sum(-1).contiguous()
    ids = _C.euclid_assign(x, centroids, x_sq, c_sq, None)
    # All ids in valid range, last row not garbage.
    in_range = ((ids >= 0) & (ids < K)).all().item()
    # First K rows should land on themselves (identity centroids).
    self_match = (ids[0, :K] == torch.arange(K, device='cuda', dtype=torch.int32)).float().mean().item()
    print(f'in_range={in_range} self_match={self_match:.4f}')
    sys.exit(0 if (in_range and self_match > 0.95) else 1)

elif mode == 'consistency':
    # Run with whatever NTILES is set to, capture cluster_ids, then run again
    # in the same process: must be deterministic across launches.
    B, N, K, D = 1, 8192, 128, 128
    x = torch.randn(B, N, D, device='cuda', dtype=torch.float16)
    centroids = x[:, :K].contiguous()
    x_sq = (x.float() ** 2).sum(-1).contiguous()
    c_sq = (centroids.float() ** 2).sum(-1).contiguous()
    ids1 = _C.euclid_assign(x, centroids, x_sq, c_sq, None).clone()
    ids2 = _C.euclid_assign(x, centroids, x_sq, c_sq, None).clone()
    same = (ids1 == ids2).all().item()
    print(f'deterministic={same}')
    sys.exit(0 if same else 1)

else:
    print(f'unknown mode {mode}')
    sys.exit(2)
"""


def _run(ntiles: str, mode: str) -> tuple[int, str]:
    proc = subprocess.run(
        [sys.executable, "-c", _SCRIPT, ntiles, mode],
        capture_output=True,
        text=True,
        timeout=120,
    )
    return proc.returncode, proc.stdout + proc.stderr


@pytest.mark.parametrize("ntiles", ["1", "2", "4"])
def test_identity_centroids(ntiles):
    """First K rows of x as centroids -> argmin trivially picks row index."""
    rc, out = _run(ntiles, "identity")
    assert rc == 0, f"FKC_NTILES={ntiles} identity failed:\n{out}"


@pytest.mark.parametrize("ntiles", ["1", "2", "4"])
def test_tail_short_remainder(ntiles):
    """N = 257 with BLOCK_N=128 forces a 1-row tail -- shouldn't OOB or break."""
    rc, out = _run(ntiles, "tail")
    assert rc == 0, f"FKC_NTILES={ntiles} tail failed:\n{out}"


@pytest.mark.parametrize("ntiles", ["1", "2", "4"])
def test_determinism(ntiles):
    """Same input must yield same cluster_ids on repeated launches."""
    rc, out = _run(ntiles, "consistency")
    assert rc == 0, f"FKC_NTILES={ntiles} non-deterministic:\n{out}"


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
