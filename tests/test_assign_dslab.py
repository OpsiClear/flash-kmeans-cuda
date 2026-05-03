# tests/test_assign_dslab.py
"""D-slab kernel correctness tests.

The smoke test (test_dslab_smoke_d256) confirms the new kernel produces correct
cluster_ids when force-routed via FKC_DSLAB=1. After Task 5 (slab inner loop),
the cross-kernel test (test_dslab_matches_narrow) confirms equivalence to the
legacy narrowk32 path for every supported D.

Each test runs in a subprocess so FKC_DSLAB takes effect at first launch."""

from __future__ import annotations

import subprocess
import sys

import pytest


def _run_dslab_script(D: int, K: int, dtype: str = "float16") -> tuple[int, str]:
    script = f"""
import os
os.environ['FKC_DSLAB'] = '1'
import torch
from flash_kmeans_cuda import _C

torch.manual_seed(0)
B, N, D, K = 1, 2048, {D}, {K}
dtype = torch.{dtype}
x = torch.randn(B, N, D, device='cuda', dtype=dtype)
centroids = x[:, :K].contiguous()
x_sq = (x.float() ** 2).sum(-1).contiguous()
c_sq = (centroids.float() ** 2).sum(-1).contiguous()

ids = _C.euclid_assign(x, centroids, x_sq, c_sq, None)

# Python fp32 reference.
diff = x.unsqueeze(2).float() - centroids.unsqueeze(1).float()
ref = (diff * diff).sum(-1).argmin(dim=-1).to(torch.int32)

disagree = (ids != ref).float().mean().item()
threshold = 0.05 if dtype == torch.bfloat16 else 0.02
assert disagree < threshold, (
    f"D={D} K={K} dtype={dtype}: {{disagree:.3%}} disagreement "
    f"(threshold {{threshold:.0%}})"
)
print(f'OK D={D} K={K} disagree={{disagree:.4%}}')
"""
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=120,
    )
    return proc.returncode, proc.stdout + proc.stderr


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_dslab_smoke_d256(dtype):
    """D=256 force-routed via FKC_DSLAB=1 produces correct cluster_ids."""
    rc, out = _run_dslab_script(D=256, K=256, dtype=dtype)
    assert rc == 0, f"D=256 dslab smoke failed:\n{out}"
