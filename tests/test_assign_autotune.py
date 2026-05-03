# tests/test_assign_autotune.py
"""Stage 3 acceptance: autotuner probes exactly once per (dtype, D, k_bucket)
cell, never on subsequent calls. Captured via FKC_AUTOTUNE_VERBOSE on stderr.

Each test runs in a subprocess so the AutotuneCache starts empty."""

from __future__ import annotations

import re
import subprocess
import sys


_SCRIPT = r"""
import os, sys
os.environ['FKC_AUTOTUNE_VERBOSE'] = '1'
import torch
from flash_kmeans_cuda import _C

torch.manual_seed(0)

def go(D, K):
    B, N = 1, 2048
    x = torch.randn(B, N, D, device='cuda', dtype=torch.float16)
    centroids = x[:, :K].contiguous()
    x_sq = (x.float() ** 2).sum(-1).contiguous()
    c_sq = (centroids.float() ** 2).sum(-1).contiguous()
    return _C.euclid_assign(x, centroids, x_sq, c_sq, None)

# First sweep: every (D, k_bucket) cell touched here should print probes.
for D in [64, 128, 256]:
    for K in [64, 256, 1024]:
        go(D, K)

print('--- second pass ---', file=sys.stderr, flush=True)

# Second sweep: same shapes -> ZERO new probe lines.
for D in [64, 128, 256]:
    for K in [64, 256, 1024]:
        go(D, K)
"""


def test_autotune_probes_once_per_cell():
    proc = subprocess.run(
        [sys.executable, "-c", _SCRIPT],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, f"script failed:\n{proc.stderr}"
    out = proc.stderr
    parts = out.split("--- second pass ---")
    assert len(parts) == 2, f"missing marker; full stderr:\n{out}"
    first, second = parts
    probe_re = re.compile(r"\[fkc autotune\].* probe\[\d+\]=")
    first_probes = probe_re.findall(first)
    second_probes = probe_re.findall(second)
    # First pass: each of 9 (D, K) cells * up to 3 candidates = up to 27
    # probe lines. Lower bound: each cell prints >= 1 candidate.
    assert len(first_probes) >= 9, (
        f"expected >=9 probes on first pass, got {len(first_probes)}\n{first}"
    )
    # Second pass: zero new probes (cache hits everywhere).
    assert len(second_probes) == 0, (
        f"expected zero probes on second pass, got {len(second_probes)}\n{second}"
    )


def test_autotune_disabled_uses_static_order():
    """FKC_AUTOTUNE=0 must skip the probe entirely (no probe log lines)."""
    script = (
        "import os; os.environ['FKC_AUTOTUNE'] = '0'; "
        "os.environ['FKC_AUTOTUNE_VERBOSE'] = '1'\n"
        "import torch\n"
        "from flash_kmeans_cuda import _C\n"
        "torch.manual_seed(0)\n"
        "x = torch.randn(1, 2048, 128, device='cuda', dtype=torch.float16)\n"
        "centroids = x[:, :256].contiguous()\n"
        "x_sq = (x.float() ** 2).sum(-1).contiguous()\n"
        "c_sq = (centroids.float() ** 2).sum(-1).contiguous()\n"
        "for _ in range(5):\n"
        "    _C.euclid_assign(x, centroids, x_sq, c_sq, None)\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert "[fkc autotune]" not in proc.stderr, (
        f"expected no probe lines with FKC_AUTOTUNE=0, got:\n{proc.stderr}"
    )


def test_autotune_concurrent_first_call_single_probe():
    """Two host threads launching assign on the same shape concurrently must
    cause exactly one probe (the second thread sees probed=true and reuses).

    Run in a subprocess to keep VERBOSE output clean.
    """
    script = r"""
import os
os.environ['FKC_AUTOTUNE_VERBOSE'] = '1'
import threading
import torch
from flash_kmeans_cuda import _C

torch.manual_seed(0)
B, N, K, D = 1, 2048, 256, 128
x = torch.randn(B, N, D, device='cuda', dtype=torch.float16)
centroids = x[:, :K].contiguous()
x_sq = (x.float() ** 2).sum(-1).contiguous()
c_sq = (centroids.float() ** 2).sum(-1).contiguous()

barrier = threading.Barrier(4)

def worker():
    barrier.wait()
    for _ in range(3):
        _C.euclid_assign(x, centroids, x_sq, c_sq, None)

threads = [threading.Thread(target=worker) for _ in range(4)]
for t in threads: t.start()
for t in threads: t.join()
torch.cuda.synchronize()
"""
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=180,
    )
    assert proc.returncode == 0, proc.stderr
    # Expect probe lines for ONE (D=128, k_bucket=1) cell only — exactly
    # min(3, n_feasible) lines, which for D=128 mid-K is 3.
    probes = re.findall(r"\[fkc autotune\].* probe\[\d+\]=", proc.stderr)
    cells = set(
        re.findall(r"D_idx=(\d+) k_bucket=(\d+)", proc.stderr)
    )
    assert len(cells) == 1, (
        f"expected exactly 1 cell probed under concurrent load, got "
        f"{len(cells)} cells: {cells}\n{proc.stderr}"
    )
    assert 1 <= len(probes) <= 3, (
        f"expected 1-3 probe lines for one cell, got {len(probes)}:\n{proc.stderr}"
    )
